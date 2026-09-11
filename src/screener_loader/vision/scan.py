"""Vision scan runner: freeze, render, batch, retry, resume. No source adapter."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import date, datetime, timezone
from threading import RLock
from typing import Any, Callable, Sequence
from uuid import uuid4
import logging
import random as random_mod
import time

from .batching import retry_delay_seconds, sort_candidates, split_candidates
from .protocols import CancelFlag
from .types import (
    AttemptError,
    CandidateInput,
    CandidateResult,
    ChartArtifact,
    ClassificationAttempt,
    CompiledRequest,
    FeatureValue,
    FrozenInputConflictError,
    OversizeRequestError,
    PreparedScan,
    RunSummary,
    ScanOutcome,
    SemanticValidationError,
    TokenUsage,
    VisionError,
)
from .validation import validate_classification_payload

log = logging.getLogger("screener_loader.vision.scan")

SleepFn = Callable[[float], None]
RandomFn = Callable[[], float]


class _StopDispatch(Exception):
    def __init__(self, error: AttemptError) -> None:
        super().__init__(error.message)
        self.error = error


def _cancelled(cancel: CancelFlag | None) -> bool:
    return bool(cancel is not None and cancel.is_set())


def _sum_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    known = [u for u in usages if u is not None]
    if not known:
        return None
    return TokenUsage(
        input_tokens=sum(u.input_tokens or 0 for u in known) or None,
        output_tokens=sum(u.output_tokens or 0 for u in known) or None,
        total_tokens=sum(u.total_tokens or 0 for u in known) or None,
    )


def _payload_from_results(results: Sequence[Any]) -> dict[str, Any]:
    rows = []
    for item in results:
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "assessments": [
                    {
                        "setup_id": a.setup_id,
                        "verdict": a.verdict,
                        "match_strength": a.match_strength,
                        "reason": a.reason,
                        "violated_required_rule_ids": list(a.violated_required_rule_ids),
                        "missing_evidence": list(a.missing_evidence),
                    }
                    for a in item.assessments
                ],
            }
        )
    return {"results": rows}


class Scanner:
    """Injected-interface runner. Accepts PreparedScan only."""

    def __init__(
        self,
        *,
        renderer,
        compiler,
        classifier,
        store,
        sleep: SleepFn | None = None,
        random: RandomFn | None = None,
    ) -> None:
        self.renderer = renderer
        self.compiler = compiler
        self.classifier = classifier
        self.store = store
        self._sleep = time.sleep if sleep is None else sleep
        self._random = random_mod.random if random is None else random
        self.peak_in_flight = 0
        self._in_flight = 0
        self._write_lock = RLock()
        self._usages: list[TokenUsage | None] = []
        self._attempts = 0
        self._stop_error: AttemptError | None = None

    def run(self, prepared: PreparedScan, *, cancel: CancelFlag | None = None) -> ScanOutcome:
        self._preflight(prepared)
        stored = self.store.create_run(prepared)
        with self.store.lock_run(stored.run_id):
            return self._execute(stored.run_id, prepared, cancel)

    def resume(
        self,
        run_id: str,
        prepared: PreparedScan | None = None,
        *,
        cancel: CancelFlag | None = None,
    ) -> ScanOutcome:
        with self.store.lock_run(run_id):
            frozen = self.store.load_frozen_inputs(run_id)
            if prepared is not None and prepared.semantic_digest != frozen.semantic_digest:
                raise FrozenInputConflictError(
                    "Prepared scan semantic digest does not match the frozen run; start a new run"
                )
            self._preflight(frozen)
            return self._execute(run_id, frozen, cancel)

    def _preflight(self, prepared: PreparedScan) -> None:
        if not prepared.config.model.strip():
            raise VisionError("ScanConfig.model must be an explicit model id")
        setup_ids = {s.setup_id for s in prepared.setups}
        for ex in prepared.examples:
            if ex.type == "image" and not ex.image_bytes:
                raise VisionError(f"Required image example {ex.scoped_id} is missing bytes")
            if ex.type == "market_window" and ex.window is None:
                raise VisionError(f"Required market_window example {ex.scoped_id} is missing bars")
        for cand in prepared.candidates:
            if cand.profile != prepared.profile:
                raise VisionError(f"Candidate {cand.candidate_id} profile disagrees with the run profile")
            missing = [s for s in cand.eligible_setup_ids if s not in setup_ids]
            if missing:
                raise VisionError(f"Candidate {cand.candidate_id} lists unknown setups {missing}")
            if not cand.eligible_setup_ids:
                raise VisionError(f"Candidate {cand.candidate_id} has no eligible setups")
            if cand.source_digest != cand.window.source_digest:
                raise VisionError(f"Candidate {cand.candidate_id} source digest drifted from its window")
        self._preflight_request_limits(prepared)

    def _preflight_request_limits(self, prepared: PreparedScan) -> None:
        from .prompts import estimate_output_tokens

        if not prepared.candidates:
            return
        config = prepared.config
        ref_counts: dict[str, int] = {}
        for ex in prepared.examples:
            ref_counts[ex.setup_id] = ref_counts.get(ex.setup_id, 0) + 1
        ref_total = sum(ref_counts.values())
        limit = int(config.max_images_per_request)
        detail = ", ".join(f"{sid}={n} example(s)" for sid, n in sorted(ref_counts.items())) or "none"
        if ref_total > limit:
            raise OversizeRequestError(
                f"Reference images alone are {ref_total} ({detail}); "
                f"max_images_per_request={limit}. Reduce examples or raise the limit. "
                "Splitting candidates cannot fix this."
            )
        if ref_total + 1 > limit:
            raise OversizeRequestError(
                f"Reference images are {ref_total} ({detail}), leaving no room for a candidate chart "
                f"under max_images_per_request={limit}."
            )
        for cand in prepared.candidates:
            est = estimate_output_tokens(len(cand.eligible_setup_ids))
            if est > int(config.max_output_tokens):
                raise OversizeRequestError(
                    f"Candidate {cand.candidate_id} alone is estimated at {est} output tokens, "
                    f"exceeding max_output_tokens={config.max_output_tokens}. "
                    "Raise the cap or reduce eligible setups; splitting cannot help."
                )

    def _request_compiler(self, prepared: PreparedScan):
        from .prompts import SnapshotRequestCompiler

        snap = prepared.compiler
        if snap is not None and isinstance(self.compiler, SnapshotRequestCompiler):
            return SnapshotRequestCompiler.from_snapshot(snap)
        return self.compiler

    def _execute(self, run_id: str, prepared: PreparedScan, cancel: CancelFlag | None) -> ScanOutcome:
        self._usages = []
        self._attempts = 0
        self._stop_error = None
        self.peak_in_flight = 0
        self._in_flight = 0
        dry = prepared.config.mode == "dry_run"
        synthetic = prepared.config.mode == "demo" or getattr(self.classifier, "provider", "") == "fake"
        if dry:
            self.store.update_status(run_id, "dry_run")

        self._record_skips(run_id, prepared)
        example_artifacts = self._prepare_references(run_id, prepared)

        committed = {
            r.candidate_id
            for r in self.store.list_candidate_results(run_id)
            if r.status in {"completed", "skipped"}
        }
        pending = [c for c in sort_candidates(prepared.candidates) if c.candidate_id not in committed]
        if _cancelled(cancel):
            return self._finalize(run_id, prepared, status="cancelled", synthetic=synthetic)

        if not pending:
            status = "dry_run" if dry else self._status_from_store(run_id, prepared)
            return self._finalize(run_id, prepared, status=status, synthetic=synthetic)

        try:
            self._run_queue(run_id, prepared, pending, example_artifacts, cancel, dry=dry)
        except _StopDispatch as stop:
            self._stop_error = stop.error
            log.info("stopping dispatch: %s", stop.error.kind)

        if _cancelled(cancel):
            status = "cancelled"
        elif dry:
            status = "dry_run"
        else:
            status = self._status_from_store(run_id, prepared)
        return self._finalize(run_id, prepared, status=status, synthetic=synthetic)

    def _run_queue(
        self,
        run_id: str,
        prepared: PreparedScan,
        pending: list[CandidateInput],
        example_artifacts: tuple[ChartArtifact, ...],
        cancel: CancelFlag | None,
        *,
        dry: bool,
    ) -> None:
        config = prepared.config
        max_in_flight = min(int(config.max_concurrency), int(config.max_in_flight_batches))
        remaining = list(pending)
        with ThreadPoolExecutor(max_workers=max_in_flight) as pool:
            futures: dict[Any, tuple[CandidateInput, ...]] = {}

            def submit_more() -> None:
                while remaining and len(futures) < max_in_flight and not _cancelled(cancel) and self._stop_error is None:
                    take = min(int(config.batch_size), len(remaining))
                    batch = tuple(remaining[:take])
                    del remaining[:take]
                    fut = pool.submit(
                        self._process_batch,
                        run_id,
                        prepared,
                        batch,
                        example_artifacts,
                        cancel,
                        dry,
                        0,
                    )
                    futures[fut] = batch
                    self._in_flight = len(futures)
                    self.peak_in_flight = max(self.peak_in_flight, self._in_flight)

            submit_more()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut)
                    self._in_flight = len(futures)
                    exc = fut.exception()
                    if isinstance(exc, _StopDispatch):
                        self._stop_error = exc.error
                        remaining.clear()
                    elif exc is not None:
                        raise exc
                if self._stop_error is None and not _cancelled(cancel):
                    submit_more()
                elif remaining:
                    remaining.clear()

    def _process_batch(
        self,
        run_id: str,
        prepared: PreparedScan,
        batch: tuple[CandidateInput, ...],
        example_artifacts: tuple[ChartArtifact, ...],
        cancel: CancelFlag | None,
        dry: bool,
        depth: int,
    ) -> None:
        if _cancelled(cancel) or self._stop_error is not None:
            return
        with self._write_lock:
            committed = {
                r.candidate_id
                for r in self.store.list_candidate_results(run_id)
                if r.status in {"completed", "skipped"}
            }
        work = tuple(c for c in batch if c.candidate_id not in committed)
        if not work:
            return

        rendered: list[tuple[CandidateInput, ChartArtifact]] = []
        render_failures: list[CandidateResult] = []
        for cand in work:
            try:
                art = self.renderer.render(
                    cand.window,
                    prepared.profile,
                    title=f"{cand.ticker}  {cand.asof_date.isoformat()}",
                )
                art = replace(art, kind="candidate", candidate_id=cand.candidate_id, ticker=cand.ticker)
                with self._write_lock:
                    self.store.save_artifact(run_id, art)
                rendered.append((cand, art))
            except Exception as exc:
                render_failures.append(
                    self._candidate_error(
                        cand,
                        AttemptError(
                            kind="unavailable_input",
                            message=str(exc)[:300],
                            retryable=False,
                        ),
                    )
                )
        if render_failures:
            with self._write_lock:
                self.store.mark_candidates(run_id, render_failures)
        if not rendered:
            return

        ok_cands = tuple(c for c, _ in rendered)
        ok_arts = tuple(a for _, a in rendered)
        batch_id = f"batch-{uuid4().hex[:10]}"
        try:
            request = self._request_compiler(prepared).compile(
                setups=prepared.setups,
                example_artifacts=example_artifacts,
                examples=prepared.examples,
                candidate_artifacts=ok_arts,
                candidates=ok_cands,
                config=prepared.config,
                batch_id=batch_id,
            )
        except OversizeRequestError as exc:
            if len(ok_cands) > 1 and depth < 8:
                left, right = split_candidates(ok_cands)
                self._process_batch(run_id, prepared, left, example_artifacts, cancel, dry, depth + 1)
                if right:
                    self._process_batch(run_id, prepared, right, example_artifacts, cancel, dry, depth + 1)
                return
            err = AttemptError(kind="oversize", message=str(exc)[:400], retryable=False)
            with self._write_lock:
                self.store.mark_candidates(run_id, [self._candidate_error(c, err) for c in ok_cands])
            return

        if dry:
            attempt = self._dry_attempt(request)
            self._journal(run_id, attempt)
            self._record_dry_charts(run_id, tuple(zip(ok_cands, ok_arts, strict=True)), attempt)
            return

        if _cancelled(cancel):
            return

        recovered = self.store.recover_attempt(run_id, request.fingerprint)
        if recovered is not None and recovered.error is None and (
            recovered.results is not None or recovered.sanitized_output is not None
        ):
            attempt = recovered
        else:
            attempt = self._classify_with_retries(run_id, prepared, request, ok_cands, cancel)

        if attempt.error and attempt.error.kind in {"auth", "config"}:
            self._mark_errors(run_id, ok_cands, attempt)
            raise _StopDispatch(attempt.error)

        if attempt.accepted or attempt.results is not None or attempt.sanitized_output is not None:
            try:
                payload = (
                    _payload_from_results(attempt.results)
                    if attempt.results is not None
                    else dict(attempt.sanitized_output or {})
                )
                validated = validate_classification_payload(
                    payload, candidates=ok_cands, setups=prepared.setups
                )
                attempt = replace(attempt, results=validated, accepted=True, error=None)
                self._commit(run_id, request, ok_cands, attempt)
                return
            except SemanticValidationError as exc:
                attempt = replace(
                    attempt,
                    accepted=False,
                    error=AttemptError(kind="invalid_output", message=str(exc)[:400], retryable=len(ok_cands) > 1),
                    results=None,
                )
                self._journal(run_id, attempt)
                if len(ok_cands) > 1 and depth < 8:
                    left, right = split_candidates(ok_cands)
                    self._process_batch(run_id, prepared, left, example_artifacts, cancel, dry, depth + 1)
                    if right:
                        self._process_batch(run_id, prepared, right, example_artifacts, cancel, dry, depth + 1)
                    return

        if attempt.error and attempt.error.kind in {"incomplete", "oversize"} and len(ok_cands) > 1 and depth < 8:
            left, right = split_candidates(ok_cands)
            self._process_batch(run_id, prepared, left, example_artifacts, cancel, dry, depth + 1)
            if right:
                self._process_batch(run_id, prepared, right, example_artifacts, cancel, dry, depth + 1)
            return

        self._mark_errors(run_id, ok_cands, attempt)

    def _classify_with_retries(
        self,
        run_id: str,
        prepared: PreparedScan,
        request: CompiledRequest,
        candidates: tuple[CandidateInput, ...],
        cancel: CancelFlag | None,
    ) -> ClassificationAttempt:
        last: ClassificationAttempt | None = None
        retries = int(prepared.config.max_retries)
        for i in range(retries + 1):
            if _cancelled(cancel):
                err = AttemptError(kind="cancelled", message="cancelled before classify", retryable=False)
                return self._error_attempt(request, err)
            attempt = self.classifier.classify(request)
            with self._write_lock:
                self._attempts += 1
                self._usages.append(attempt.usage)
            self._journal(run_id, attempt)
            last = attempt
            if attempt.error is None:
                return attempt
            if attempt.error.kind in {"auth", "config", "refusal"}:
                return attempt
            if not attempt.error.retryable or i >= retries:
                return attempt
            delay = retry_delay_seconds(
                i,
                prepared.config,
                retry_after=attempt.error.retry_after_seconds or attempt.retry_after_seconds,
                rng_random=self._random(),
            )
            self._sleep(delay)
        assert last is not None
        return last

    def _prepare_references(self, run_id: str, prepared: PreparedScan) -> tuple[ChartArtifact, ...]:
        out: list[ChartArtifact] = []
        for ex in prepared.examples:
            if ex.type == "image":
                art = self.renderer.wrap_upload(ex)
            else:
                if ex.window is None:
                    raise VisionError(f"Required market_window example {ex.scoped_id} is missing bars")
                art = self.renderer.render(
                    ex.window,
                    prepared.profile,
                    title=f"Example {ex.scoped_id} ({ex.polarity})",
                )
                art = replace(
                    art,
                    kind="example",
                    setup_id=ex.setup_id,
                    example_id=ex.example_id,
                    candidate_id=None,
                    source="rendered",
                )
            with self._write_lock:
                self.store.save_artifact(run_id, art)
            out.append(art)
        return tuple(out)

    def _record_skips(self, run_id: str, prepared: PreparedScan) -> None:
        if not prepared.skips:
            return
        rows = []
        for skip in prepared.skips:
            rows.append(
                CandidateResult(
                    candidate_id=f"skip:{skip.ticker}:{skip.kind}",
                    status="skipped",
                    ticker=skip.ticker,
                    asof_date=skip.asof_date or date(1970, 1, 1),
                    features=skip.features if skip.features is not None else FeatureValue(None, None, None),
                    eligible_setup_ids=(),
                    assessments=(),
                    artifact_id=None,
                    error=AttemptError(kind=skip.kind, message=skip.message, retryable=False),  # type: ignore[arg-type]
                    attempt_id=None,
                    source_digest="",
                )
            )
        with self._write_lock:
            self.store.mark_candidates(run_id, rows)

    def _record_dry_charts(
        self,
        run_id: str,
        rendered: Sequence[tuple[CandidateInput, ChartArtifact]],
        attempt: ClassificationAttempt,
    ) -> None:
        rows = [
            CandidateResult(
                candidate_id=cand.candidate_id,
                status="completed",
                ticker=cand.ticker,
                asof_date=cand.asof_date,
                features=cand.features,
                eligible_setup_ids=cand.eligible_setup_ids,
                assessments=(),
                artifact_id=art.artifact_id,
                error=None,
                attempt_id=attempt.attempt_id,
                source_digest=cand.source_digest,
            )
            for cand, art in rendered
        ]
        with self._write_lock:
            self.store.mark_candidates(run_id, rows)

    def _commit(
        self,
        run_id: str,
        request: CompiledRequest,
        candidates: Sequence[CandidateInput],
        attempt: ClassificationAttempt,
    ) -> None:
        by_id = {c.candidate_id: c for c in candidates}
        rows = []
        for item in attempt.results or ():
            cand = by_id[item.candidate_id]
            rows.append(
                CandidateResult(
                    candidate_id=cand.candidate_id,
                    status="completed",
                    ticker=cand.ticker,
                    asof_date=cand.asof_date,
                    features=cand.features,
                    eligible_setup_ids=cand.eligible_setup_ids,
                    assessments=item.assessments,
                    artifact_id=f"candidate:{cand.ticker}:{cand.asof_date.isoformat()}",
                    error=None,
                    attempt_id=attempt.attempt_id,
                    source_digest=cand.source_digest,
                )
            )
        with self._write_lock:
            self.store.commit_batch(run_id, batch_id=request.batch_id, results=rows, attempt=attempt)

    def _mark_errors(self, run_id: str, candidates: Sequence[CandidateInput], attempt: ClassificationAttempt) -> None:
        err = attempt.error or AttemptError(kind="provider", message="classification failed", retryable=False)
        rows = [self._candidate_error(c, err, attempt_id=attempt.attempt_id) for c in candidates]
        with self._write_lock:
            self.store.mark_candidates(run_id, rows)

    def _candidate_error(self, cand: CandidateInput, error: AttemptError, *, attempt_id: str | None = None) -> CandidateResult:
        return CandidateResult(
            candidate_id=cand.candidate_id,
            status="error",
            ticker=cand.ticker,
            asof_date=cand.asof_date,
            features=cand.features,
            eligible_setup_ids=cand.eligible_setup_ids,
            assessments=(),
            artifact_id=None,
            error=error,
            attempt_id=attempt_id,
            source_digest=cand.source_digest,
        )

    def _journal(self, run_id: str, attempt: ClassificationAttempt) -> None:
        with self._write_lock:
            self.store.journal_attempt(run_id, attempt)

    def _stored_attempts(self, run_id: str) -> tuple[ClassificationAttempt, ...]:
        if hasattr(self.store, "list_attempts"):
            return tuple(self.store.list_attempts(run_id))
        attempts = getattr(self.store, "attempts", None)
        if isinstance(attempts, dict):
            return tuple(attempts.get(run_id, ()))
        return ()

    def _aggregate_usage(self, run_id: str) -> tuple[int, TokenUsage | None]:
        from .store import aggregate_attempt_usage

        return aggregate_attempt_usage(self._stored_attempts(run_id))

    def _dry_attempt(self, request: CompiledRequest) -> ClassificationAttempt:
        now = datetime.now(timezone.utc)
        return ClassificationAttempt(
            attempt_id=str(uuid4()),
            batch_id=request.batch_id,
            batch_fingerprint=request.fingerprint,
            candidate_ids=request.candidate_ids,
            setup_ids=request.setup_ids,
            started_at=now,
            ended_at=now,
            provider="none",
            model=request.model,
            response_id=None,
            usage=None,
            retry_after_seconds=None,
            latency_ms=0,
            error=None,
            sanitized_output={
                "dry_run": True,
                "estimates": {
                    "transport_bytes_estimate": request.estimates.transport_bytes_estimate,
                    "notes": request.estimates.notes,
                },
            },
            results=None,
            accepted=False,
        )

    def _error_attempt(self, request: CompiledRequest, error: AttemptError) -> ClassificationAttempt:
        now = datetime.now(timezone.utc)
        return ClassificationAttempt(
            attempt_id=str(uuid4()),
            batch_id=request.batch_id,
            batch_fingerprint=request.fingerprint,
            candidate_ids=request.candidate_ids,
            setup_ids=request.setup_ids,
            started_at=now,
            ended_at=now,
            provider=getattr(self.classifier, "provider", "unknown"),
            model=request.model,
            response_id=None,
            usage=None,
            retry_after_seconds=error.retry_after_seconds,
            latency_ms=0,
            error=error,
            sanitized_output=None,
            results=None,
            accepted=False,
        )

    def _status_from_store(self, run_id: str, prepared: PreparedScan) -> str:
        rows = self.store.list_candidate_results(run_id)
        real_ids = {c.candidate_id for c in prepared.candidates}
        real = [r for r in rows if r.candidate_id in real_ids]
        completed = sum(1 for r in real if r.status == "completed")
        errors = sum(1 for r in real if r.status == "error")
        if self._stop_error and self._stop_error.kind in {"auth", "config"} and completed == 0:
            return "failed"
        if errors and completed:
            return "partial"
        if errors and completed == 0:
            return "failed"
        if completed + sum(1 for r in real if r.status == "skipped") >= len(prepared.candidates) and errors == 0:
            return "completed"
        if completed:
            return "partial"
        return "failed"

    def _finalize(self, run_id: str, prepared: PreparedScan, *, status: str, synthetic: bool) -> ScanOutcome:
        rows = self.store.list_candidate_results(run_id)
        real_ids = {c.candidate_id for c in prepared.candidates}
        real = [r for r in rows if r.candidate_id in real_ids]
        skipped_ids = {r.candidate_id for r in rows if r.status == "skipped"}
        completed = sum(1 for r in real if r.status == "completed")
        errors = sum(1 for r in real if r.status == "error")
        skipped = len(skipped_ids)
        pending = 0
        by_id = {r.candidate_id: r for r in real}
        for cand in prepared.candidates:
            row = by_id.get(cand.candidate_id)
            if row is None or row.status == "pending":
                pending += 1
        matches = sum(sum(1 for a in r.assessments if a.verdict == "match") for r in real)
        attempts, usage = self._aggregate_usage(run_id)
        summary = RunSummary(
            candidates_total=len(prepared.candidates) + len(prepared.skips),
            candidates_completed=completed,
            candidates_error=errors,
            candidates_skipped=skipped,
            candidates_pending=pending,
            setup_matches=matches,
            attempts=attempts,
            usage=usage,
            status=status,  # type: ignore[arg-type]
            synthetic=synthetic,
        )
        self.store.finalize(run_id, summary)
        return ScanOutcome(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            summary=summary,
            diagnostics=prepared.diagnostics,
            semantic_digest=prepared.semantic_digest,
            raw_digest=prepared.raw_digest,
        )


# Public alias matching the contract name.
VisionScanner = Scanner
