"""Shared vision fixtures and protocol fakes. Does not import scan/client/charts implementations."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
import struct
import zlib
from typing import Any, Sequence
from uuid import uuid4

from screener_loader.setups.spec import (
    ChartStyle,
    SetupCriteria,
    SetupFilters,
    SetupSpec,
    VisionExample,
)
from screener_loader.vision.serialization import digest_bytes
from screener_loader.vision.snapshots import (
    bars_from_rows,
    feature_value_from_mapping,
    freeze_prepared_scan,
    make_bar_window,
    make_candidate_input,
    snapshot_example,
    snapshot_setup,
)
from screener_loader.vision.types import (
    ArtifactRef,
    Assessment,
    AttemptError,
    CandidateClassification,
    CandidateDetail,
    CandidateInput,
    CandidateResult,
    CandidateRow,
    ChartArtifact,
    ChartProfile,
    ClassificationAttempt,
    CompiledRequest,
    ExampleInput,
    InputSkip,
    PreparedScan,
    RenderedImage,
    ResultCounts,
    ResultPage,
    ResultQuery,
    ReviewRecord,
    RunSummary,
    ScanConfig,
    ScanDiagnostics,
    SetupSnapshot,
    StoredRun,
    TokenUsage,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "vision"


def synthetic_png_bytes(width: int = 8, height: int = 6, color: tuple[int, int, int] = (20, 80, 160)) -> bytes:
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


SYNTHETIC_PNG = synthetic_png_bytes()
SYNTHETIC_PNG_PATH = FIXTURES_DIR / "synthetic.png"


def write_synthetic_png(path: Path | None = None) -> Path:
    dest = path or SYNTHETIC_PNG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(SYNTHETIC_PNG)
    return dest


def make_spec(
    setup_id: str,
    *,
    name: str | None = None,
    lookback_bars: int = 100,
    volume: bool = True,
    moving_averages: tuple[int, ...] = (10, 20, 50),
    required: tuple[str, ...] = ("tight flag",),
    preferred: tuple[str, ...] = ("volume dry-up",),
    disqualifiers: tuple[str, ...] = ("too extended",),
    min_adr_pct_20: float | None = None,
    min_market_cap: float | None = None,
    enabled: bool = True,
) -> SetupSpec:
    return SetupSpec(
        id=setup_id,
        name=name or setup_id.replace("_", " ").title(),
        enabled=enabled,
        lookback_bars=lookback_bars,
        description="Controlled consolidation after an impulse.",
        criteria=SetupCriteria(required=required, preferred=preferred, disqualifiers=disqualifiers),
        filters=SetupFilters(min_adr_pct_20=min_adr_pct_20, min_market_cap=min_market_cap),
        chart=ChartStyle(volume=volume, moving_averages=moving_averages),
    )


def make_example(
    example_id: str,
    *,
    polarity: str = "positive",
    quality: str | None = "canonical",
    note: str = "",
    ticker: str = "NVDA",
    asof: date | None = None,
    kind: str = "market_window",
) -> VisionExample:
    if kind == "image":
        return VisionExample(
            id=example_id,
            polarity=polarity,  # type: ignore[arg-type]
            type="image",
            quality=quality,  # type: ignore[arg-type]
            note=note,
            path="examples/synthetic.png",
        )
    return VisionExample(
        id=example_id,
        polarity=polarity,  # type: ignore[arg-type]
        type="market_window",
        quality=quality,  # type: ignore[arg-type]
        note=note,
        ticker=ticker,
        date=asof or date(2026, 6, 18),
    )


def yaml_bytes_for(spec: SetupSpec) -> bytes:
    # Deterministic stand-in for original setup.yaml bytes (Agent 3 supplies real file bytes).
    return (
        f"id: {spec.id}\nname: {spec.name}\nlookback_bars: {spec.lookback_bars}\n"
        f"chart.volume: {str(spec.chart.volume).lower()}\n"
        f"chart.moving_averages: {list(spec.chart.moving_averages)}\n"
    ).encode("utf-8")


def ohlcv_rows(
    ticker: str,
    *,
    asof: date = date(2026, 6, 18),
    n: int = 30,
    close0: float = 100.0,
    volume: float = 1_000_000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = asof - timedelta(days=n - 1)
    for i in range(n):
        d = start + timedelta(days=i)
        px = close0 + i * 0.4
        rows.append(
            {
                "ticker": ticker,
                "date": d,
                "open": px - 0.2,
                "high": px + 0.8,
                "low": px - 0.7,
                "close": px,
                "volume": volume + i,
            }
        )
    return rows


def window_for(ticker: str, *, n: int = 30, asof: date = date(2026, 6, 18), provenance: str = "unknown"):
    bars = bars_from_rows(ohlcv_rows(ticker, asof=asof, n=n))
    return make_bar_window(ticker, bars, provenance=provenance)


def default_profile(**kwargs: Any) -> ChartProfile:
    payload = {
        "timeframe": "1d",
        "lookback_bars": 100,
        "volume": True,
        "moving_averages": (10, 20, 50),
    }
    payload.update(kwargs)
    return ChartProfile(**payload)  # type: ignore[arg-type]


def snapshot_pair(
    spec: SetupSpec,
    examples: Sequence[VisionExample],
    *,
    image_bytes: bytes | None = None,
    windows: dict[str, Any] | None = None,
) -> tuple[SetupSnapshot, tuple[ExampleInput, ...]]:
    snap = snapshot_setup(spec, yaml_bytes=yaml_bytes_for(spec), examples=examples)
    out: list[ExampleInput] = []
    wins = windows or {}
    for ex in examples:
        if ex.type == "image":
            out.append(snapshot_example(spec, ex, image_bytes=image_bytes or SYNTHETIC_PNG))
        else:
            win = wins.get(ex.id) or window_for(ex.ticker or "NVDA", n=min(spec.lookback_bars, 40), asof=ex.date)
            out.append(snapshot_example(spec, ex, window=win))
    return snap, tuple(out)


def sample_prepared_scan(
    *,
    lookback_bars: int = 100,
    candidates: Sequence[CandidateInput] | None = None,
    skips: Sequence[InputSkip] = (),
    diagnostics: ScanDiagnostics | None = None,
    mode: str = "live",
    adr: float | None = 0.05,
) -> PreparedScan:
    spec = make_spec("flag", lookback_bars=lookback_bars)
    examples = [
        make_example("ex_canonical", quality="canonical", polarity="positive"),
        make_example("ex_near", quality="near_miss", polarity="negative", ticker="XYZ"),
        make_example("ex_decent", quality="decent", polarity="positive", ticker="AAPL"),
        make_example("upload_1", kind="image", polarity="negative", quality="edge_case"),
    ]
    setup, example_inputs = snapshot_pair(spec, examples)
    profile = default_profile(lookback_bars=lookback_bars)
    feats = feature_value_from_mapping(
        {"close": 12.5, "dollar_vol_avg_20": 8_000_000.0, "adr_pct_20": adr}
    )
    if candidates is None:
        win = window_for("AAPL", n=min(lookback_bars, 40))
        candidates = (
            make_candidate_input(
                ticker="AAPL",
                window=win,
                features=feats,
                eligible_setup_ids=("flag",),
                profile=profile,
            ),
        )
    return freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=example_inputs,
        candidates=tuple(candidates),
        config=ScanConfig(model="gpt-4.1-2025-04-14", mode=mode),  # type: ignore[arg-type]
        diagnostics=diagnostics or ScanDiagnostics(universe_count=3, input_count=2, eligible_union_count=1),
        skips=skips,
        global_filters_yaml_bytes=b"min_price: 2.0\nmin_dollar_vol_20d: 5000000\n",
    )


def assessment(
    setup_id: str,
    verdict: str,
    *,
    strength: int | None = None,
    reason: str = "Price coiled under resistance with drying volume.",
    violated: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> Assessment:
    return Assessment(
        setup_id=setup_id,
        verdict=verdict,  # type: ignore[arg-type]
        match_strength=strength,
        reason=reason,
        violated_required_rule_ids=violated,
        missing_evidence=missing,
    )


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 6, 19, 14, 0, tzinfo=timezone.utc)
        self.sleeps: list[float] = []

    def time(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now = self.now + timedelta(seconds=float(seconds))


class FakeRng:
    def __init__(self, values: Sequence[float] | None = None) -> None:
        self._values = list(values or [0.0])
        self._i = 0

    def random(self) -> float:
        v = self._values[self._i % len(self._values)]
        self._i += 1
        return float(v)


class FakeRenderer:
    def render(self, window, profile, *, title: str | None = None):
        png = SYNTHETIC_PNG
        img = RenderedImage(
            png_bytes=png,
            width=8,
            height=6,
            sha256=digest_bytes(png),
        )
        return ChartArtifact(
            artifact_id=f"art-{window.ticker}-{window.asof_date}",
            kind="candidate" if title and "example" not in (title or "").lower() else "example",
            image=img,
            title=title or f"{window.ticker}  {window.asof_date}",
            profile=profile,
            ma_availability=(),
            final_session_date=window.asof_date,
            ticker=window.ticker,
            source="rendered",
        )

    def wrap_upload(self, example: ExampleInput) -> ChartArtifact:
        data = example.image_bytes or SYNTHETIC_PNG
        img = RenderedImage(png_bytes=data, width=8, height=6, sha256=digest_bytes(data))
        return ChartArtifact(
            artifact_id=f"art-{example.scoped_id}",
            kind="example",
            image=img,
            title=example.scoped_id,
            profile=None,
            ma_availability=(),
            final_session_date=example.asof_date,
            setup_id=example.setup_id,
            example_id=example.example_id,
            source="upload",
        )


class FakeCompiler:
    def __init__(self) -> None:
        self.calls = 0
        self.max_seen_candidates = 0
        self._lock = Lock()
        self.oversize_when_gt: int | None = None

    def compile(self, *, setups, example_artifacts, examples, candidate_artifacts, candidates, config, batch_id):
        from screener_loader.vision.types import OversizeRequestError, RequestBlock, RequestBudgetEstimate, load_response_schema

        cands = tuple(candidates)
        with self._lock:
            self.calls += 1
            self.max_seen_candidates = max(self.max_seen_candidates, len(cands))
        if self.oversize_when_gt is not None and len(cands) > self.oversize_when_gt:
            raise OversizeRequestError(f"oversize test {len(cands)}")
        cids = tuple(c.candidate_id for c in cands)
        sids = tuple(s.setup_id for s in setups)
        blocks = (RequestBlock(kind="text", purpose="instructions", text="assess"),)
        estimates = RequestBudgetEstimate(
            reference_image_count=len(tuple(example_artifacts)),
            candidate_image_count=len(tuple(candidate_artifacts)),
            total_image_count=len(tuple(example_artifacts)) + len(tuple(candidate_artifacts)),
            text_chars=6,
            transport_bytes_estimate=100,
            anticipated_output_tokens_estimate=256,
        )
        return CompiledRequest(
            batch_id=batch_id,
            candidate_ids=cids,
            setup_ids=sids,
            blocks=blocks,
            json_schema=load_response_schema(),
            schema_name="vision_batch_classification",
            fingerprint=f"fp-{batch_id}-{','.join(cids)}",
            estimates=estimates,
            model=config.model,
        )


class FakeClassifier:
    """Explicit synthetic classifier. Never used as a missing-key fallback."""

    provider = "fake"

    def __init__(self, script: dict[str, Sequence[Assessment]] | None = None, *, error: AttemptError | None = None) -> None:
        self.script = script or {}
        self.error = error
        self.calls = 0
        self.requests: list[CompiledRequest] = []
        self._lock = Lock()
        self._failures_remaining = 0
        self._transient_error: AttemptError | None = None

    def fail_next(self, n: int, error: AttemptError) -> None:
        self._transient_error = error
        self._failures_remaining = int(n)

    def classify(self, request: CompiledRequest) -> ClassificationAttempt:
        with self._lock:
            self.calls += 1
            self.requests.append(request)
            error = self.error
            if self._failures_remaining > 0:
                error = self._transient_error
                self._failures_remaining -= 1
        now = datetime.now(timezone.utc)
        if error is not None:
            return ClassificationAttempt(
                attempt_id=str(uuid4()),
                batch_id=request.batch_id,
                batch_fingerprint=request.fingerprint,
                candidate_ids=request.candidate_ids,
                setup_ids=request.setup_ids,
                started_at=now,
                ended_at=now,
                provider=self.provider,
                model=request.model,
                response_id=None,
                usage=TokenUsage(input_tokens=10, output_tokens=0, total_tokens=10),
                retry_after_seconds=error.retry_after_seconds,
                latency_ms=1,
                error=error,
                sanitized_output=None,
                results=None,
                accepted=False,
            )
        results = []
        for cid in request.candidate_ids:
            assessments = tuple(self.script.get(cid, ()))
            if not assessments:
                assessments = tuple(
                    assessment(sid, "no_match", reason="No coiled structure.") for sid in request.setup_ids
                )
            results.append(CandidateClassification(candidate_id=cid, assessments=assessments))
        payload = {
            "results": [
                {
                    "candidate_id": row.candidate_id,
                    "assessments": [
                        {
                            "setup_id": a.setup_id,
                            "verdict": a.verdict,
                            "match_strength": a.match_strength,
                            "reason": a.reason,
                            "violated_required_rule_ids": list(a.violated_required_rule_ids),
                            "missing_evidence": list(a.missing_evidence),
                        }
                        for a in row.assessments
                    ],
                }
                for row in results
            ]
        }
        return ClassificationAttempt(
            attempt_id=str(uuid4()),
            batch_id=request.batch_id,
            batch_fingerprint=request.fingerprint,
            candidate_ids=request.candidate_ids,
            setup_ids=request.setup_ids,
            started_at=now,
            ended_at=now,
            provider=self.provider,
            model=request.model,
            response_id=f"fake-{request.batch_id}",
            usage=TokenUsage(input_tokens=20, output_tokens=40, total_tokens=60),
            retry_after_seconds=None,
            latency_ms=1,
            error=None,
            sanitized_output=payload,
            results=tuple(results),
            accepted=False,
        )


class InMemoryRunStore:
    """Conforming RunStore fake for pipeline tests. Not a production store."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.runs: dict[str, StoredRun] = {}
        self.inputs: dict[str, PreparedScan] = {}
        self.artifacts: dict[tuple[str, str], tuple[ArtifactRef, bytes]] = {}
        self.attempts: dict[str, list[ClassificationAttempt]] = {}
        self.results: dict[str, dict[str, CandidateResult]] = {}
        self.locked: set[str] = set()
        self.export_calls = 0

    def create_run(self, prepared: PreparedScan) -> StoredRun:
        run_id = f"run-{uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        stored = StoredRun(
            run_id=run_id,
            status="dry_run" if prepared.config.mode == "dry_run" else "running",
            created_at=now,
            updated_at=now,
            semantic_digest=prepared.semantic_digest,
            raw_digest=prepared.raw_digest,
            config=prepared.config,
            profile=prepared.profile,
            synthetic=prepared.config.mode == "demo",
        )
        with self._lock:
            self.runs[run_id] = stored
            self.inputs[run_id] = prepared
            self.attempts[run_id] = []
            self.results[run_id] = {}
        return stored

    def lock_run(self, run_id: str):
        store = self

        class _Lock:
            def __enter__(self_inner):
                with store._lock:
                    if run_id in store.locked:
                        from screener_loader.vision.types import RunLockedError

                        raise RunLockedError(f"run {run_id} is locked")
                    store.locked.add(run_id)
                return self_inner

            def __exit__(self_inner, *args):
                with store._lock:
                    store.locked.discard(run_id)

        return _Lock()

    def load_run(self, run_id: str) -> StoredRun:
        return self.runs[run_id]

    def load_frozen_inputs(self, run_id: str) -> PreparedScan:
        return self.inputs[run_id]

    def save_artifact(self, run_id: str, artifact: ChartArtifact) -> ArtifactRef:
        kind = "candidates" if artifact.kind == "candidate" else "examples"
        name = artifact.candidate_id or f"{artifact.setup_id}_{artifact.example_id}" or artifact.artifact_id
        rel = f"artifacts/{kind}/{name}.png"
        ref = ArtifactRef(
            artifact_id=artifact.artifact_id,
            relative_path=rel,
            sha256=artifact.image.sha256,
            width=artifact.image.width,
            height=artifact.image.height,
            byte_length=len(artifact.image.png_bytes),
        )
        self.artifacts[(run_id, artifact.artifact_id)] = (ref, artifact.image.png_bytes)
        return ref

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes:
        return self.artifacts[(run_id, artifact_id)][1]

    def journal_attempt(self, run_id: str, attempt: ClassificationAttempt) -> None:
        self.attempts[run_id].append(attempt)

    def list_attempts(self, run_id: str) -> tuple[ClassificationAttempt, ...]:
        return tuple(self.attempts.get(run_id, ()))

    def recover_attempt(self, run_id: str, batch_fingerprint: str) -> ClassificationAttempt | None:
        found = [
            a
            for a in self.attempts.get(run_id, ())
            if a.batch_fingerprint == batch_fingerprint
            and a.error is None
            and (a.results is not None or a.sanitized_output is not None)
        ]
        return found[-1] if found else None

    def commit_batch(self, run_id: str, *, batch_id: str, results: Sequence[CandidateResult], attempt: ClassificationAttempt) -> None:
        self.journal_attempt(run_id, attempt)
        for row in results:
            prior = self.results[run_id].get(row.candidate_id)
            if prior is None or prior.status == "error":
                self.results[run_id][row.candidate_id] = row

    def mark_candidates(self, run_id: str, results: Sequence[CandidateResult]) -> None:
        for row in results:
            prior = self.results[run_id].get(row.candidate_id)
            if prior is None or prior.status == "error":
                self.results[run_id][row.candidate_id] = row

    def list_committed_candidate_ids(self, run_id: str) -> frozenset[str]:
        return frozenset(
            cid
            for cid, row in self.results[run_id].items()
            if row.status == "completed"
        )

    def list_candidate_results(self, run_id: str) -> tuple[CandidateResult, ...]:
        return tuple(self.results[run_id].values())

    def update_status(self, run_id: str, status: str) -> StoredRun:
        cur = self.runs[run_id]
        updated = StoredRun(
            run_id=cur.run_id,
            status=status,  # type: ignore[arg-type]
            created_at=cur.created_at,
            updated_at=datetime.now(timezone.utc),
            semantic_digest=cur.semantic_digest,
            raw_digest=cur.raw_digest,
            config=cur.config,
            profile=cur.profile,
            synthetic=cur.synthetic,
        )
        self.runs[run_id] = updated
        return updated

    def finalize(self, run_id: str, summary: RunSummary) -> StoredRun:
        return self.update_status(run_id, summary.status)

    def export_manifest(self, run_id: str) -> dict:
        self.export_calls += 1
        run = self.runs[run_id]
        return {
            "run_id": run_id,
            "status": run.status,
            "semantic_digest": run.semantic_digest,
            "artifacts": [ref.relative_path for ref, _ in self.artifacts.values() if True],
        }


class FakeResultReader:
    def __init__(self, store: InMemoryRunStore) -> None:
        self.store = store
        self.reviews: dict[tuple[str, str, str | None], list[ReviewRecord]] = {}

    def query(self, q: ResultQuery) -> ResultPage:
        rows = self._rows(q)
        start = (q.page.page - 1) * q.page.page_size
        chunk = rows[start : start + q.page.page_size]
        total = len(rows)
        matches = sum(len(r.matched_setups) for r in rows)
        has_next = start + q.page.page_size < total
        has_prev = q.page.page > 1
        return ResultPage(
            items=tuple(chunk),
            page=q.page.page,
            page_size=q.page.page_size,
            total_candidates=total,
            total_setup_matches=matches,
            has_next=has_next,
            has_prev=has_prev,
            next_page=(q.page.page + 1) if has_next else None,
            prev_page=(q.page.page - 1) if has_prev else None,
        )

    def _rows(self, q: ResultQuery) -> list[CandidateRow]:
        selected = set(q.setup_ids)
        out: list[CandidateRow] = []
        for row in self.store.list_candidate_results(q.run_id):
            if row.status != "completed":
                if q.verdicts:
                    continue
            assessments = row.assessments
            if selected:
                relevant = [a for a in assessments if a.setup_id in selected]
                if not relevant:
                    continue
            else:
                relevant = list(assessments)
            if q.verdicts and not any(a.verdict in q.verdicts for a in relevant):
                continue
            strengths = [a.match_strength for a in relevant if a.verdict == "match" and a.match_strength]
            if q.min_match_strength is not None or q.max_match_strength is not None:
                if not strengths:
                    continue
                best = max(strengths)
                if q.min_match_strength is not None and best < q.min_match_strength:
                    continue
                if q.max_match_strength is not None and best > q.max_match_strength:
                    continue
            else:
                best = max(strengths) if strengths else 0
            review = self.current_review(q.run_id, row.candidate_id)
            state = review.judgment if review else "unreviewed"
            if q.review_state not in {"any", None} and state != q.review_state:
                continue
            from screener_loader.vision.types import AssessmentSummary, MatchedSetupBadge

            badges = tuple(
                MatchedSetupBadge(setup_id=a.setup_id, setup_name=a.setup_id, match_strength=a.match_strength or 0)
                for a in assessments
                if a.verdict == "match"
            )
            summaries = tuple(
                AssessmentSummary(
                    setup_id=a.setup_id,
                    setup_name=a.setup_id,
                    verdict=a.verdict,
                    match_strength=a.match_strength,
                    reason=a.reason,
                )
                for a in assessments
            )
            out.append(
                (
                    best,
                    CandidateRow(
                        candidate_id=row.candidate_id,
                        ticker=row.ticker,
                        asof_date=row.asof_date,
                        status=row.status,
                        features=row.features,
                        matched_setups=badges,
                        assessments=summaries,
                        review_state=state,  # type: ignore[arg-type]
                        current_review=review,
                        chart_ref=None,
                        error=row.error,
                    ),
                )
            )
        desc = q.sort.descending if q.sort.key == "match_strength" else False
        out.sort(key=lambda item: item[1].candidate_id)
        if q.sort.key == "match_strength":
            out.sort(key=lambda item: item[0], reverse=desc)
        elif q.sort.key == "ticker":
            out.sort(key=lambda item: (item[1].ticker, item[1].candidate_id), reverse=q.sort.descending)
        return [item[1] for item in out]

    def get_detail(self, run_id: str, candidate_id: str) -> CandidateDetail:
        row = next(r for r in self.store.list_candidate_results(run_id) if r.candidate_id == candidate_id)
        from screener_loader.vision.types import AssessmentSummary, MatchedSetupBadge

        crow = CandidateRow(
            candidate_id=row.candidate_id,
            ticker=row.ticker,
            asof_date=row.asof_date,
            status=row.status,
            features=row.features,
            matched_setups=tuple(
                MatchedSetupBadge(setup_id=a.setup_id, setup_name=a.setup_id, match_strength=a.match_strength or 0)
                for a in row.assessments
                if a.verdict == "match"
            ),
            assessments=tuple(
                AssessmentSummary(
                    setup_id=a.setup_id,
                    setup_name=a.setup_id,
                    verdict=a.verdict,
                    match_strength=a.match_strength,
                    reason=a.reason,
                )
                for a in row.assessments
            ),
            review_state="unreviewed",
            current_review=None,
            chart_ref=None,
            error=row.error,
        )
        return CandidateDetail(
            row=crow,
            assessments=row.assessments,
            eligible_setup_ids=row.eligible_setup_ids,
            window=None,
            artifact=None,
            reviews=self.list_reviews(run_id, candidate_id),
            attempts=tuple(self.store.attempts.get(run_id, ())),
        )

    def counts(self, run_id: str) -> ResultCounts:
        rows = self.store.list_candidate_results(run_id)
        matches = sum(sum(1 for a in r.assessments if a.verdict == "match") for r in rows)
        return ResultCounts(
            candidates=len(rows),
            completed=sum(1 for r in rows if r.status == "completed"),
            error=sum(1 for r in rows if r.status == "error"),
            skipped=sum(1 for r in rows if r.status == "skipped"),
            pending=sum(1 for r in rows if r.status == "pending"),
            setup_matches=matches,
            unreviewed=len(rows),
            reviewed=0,
        )

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes:
        return self.store.get_artifact_bytes(run_id, artifact_id)

    def get_artifact_ref(self, run_id: str, artifact_id: str) -> ArtifactRef:
        return self.store.artifacts[(run_id, artifact_id)][0]

    def list_pages(self, q: ResultQuery) -> tuple[int, ...]:
        page = self.query(q)
        n = max(1, (page.total_candidates + q.page.page_size - 1) // q.page.page_size)
        return tuple(range(1, n + 1))

    def add_review(self, run_id: str, candidate_id: str, *, judgment: str, note: str = "", setup_id: str | None = None):
        key = (run_id, candidate_id, setup_id)
        prev = self.reviews.get(key, [])
        rec = ReviewRecord(
            review_id=str(uuid4()),
            run_id=run_id,
            candidate_id=candidate_id,
            setup_id=setup_id,
            judgment=judgment,  # type: ignore[arg-type]
            note=note,
            created_at=datetime.now(timezone.utc),
            supersedes_review_id=prev[-1].review_id if prev else None,
        )
        self.reviews.setdefault(key, []).append(rec)
        return rec

    def current_review(self, run_id: str, candidate_id: str, *, setup_id: str | None = None):
        recs = self.reviews.get((run_id, candidate_id, setup_id), [])
        return recs[-1] if recs else None

    def list_reviews(self, run_id: str, candidate_id: str) -> tuple[ReviewRecord, ...]:
        out: list[ReviewRecord] = []
        for (rid, cid, _), recs in self.reviews.items():
            if rid == run_id and cid == candidate_id:
                out.extend(recs)
        return tuple(out)
