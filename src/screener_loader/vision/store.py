"""Filesystem RunStore: JSON/Parquet shards under an injected scan-root Path.

Layout (Agent 3 binds the project-root directory):

    <scan_root>/<run_id>/
      meta.json
      lock
      frozen/...
      artifacts/candidates/{candidate_id}.png
      artifacts/examples/{setup_id}/{example_id}.png
      attempts/{attempt_id}.json
      batches/{batch_id}.json
      results/{candidate_id}.json
      reviews/{review_id}.json          (ReviewStore)
      derived/manifest.json             (rebuildable)
      derived/candidates.parquet        (convenience export)
      derived/assessments.parquet

Temporary names (``.tmp-*``, ``*.tmp``, ``*.partial``, other dotfiles) are ignored
by readers. Authoritative records are frozen inputs, attempt journals, batch shards,
per-candidate results, artifact bytes, and review files. Manifests/counters/parquet
exports are rebuilt from those records after interruption.

This module does not retry provider calls and does not invent predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4
import json
import os
import re
import tempfile

from .serialization import digest, digest_bytes
from .types import (
    ArtifactRef,
    Assessment,
    AttemptError,
    Bar,
    BarWindow,
    CandidateClassification,
    CandidateInput,
    CandidateResult,
    ChartArtifact,
    ChartProfile,
    ClassificationAttempt,
    CompilerSnapshot,
    ExampleInput,
    FeatureValue,
    FilterSnapshot,
    InputSkip,
    MaAvailability,
    PreparedScan,
    RunLockedError,
    RunSummary,
    ScanConfig,
    ScanDiagnostics,
    SetupSnapshot,
    SnapshotRule,
    StoredRun,
    TokenUsage,
    VisionError,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
RUN_DIR_PREFIX = "run-"
META_NAME = "meta.json"
LOCK_NAME = "lock"
FROZEN_NAME = "frozen.json"
ARTIFACT_INDEX_NAME = "index.json"

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_temp_name(name: str) -> bool:
    if not name or name in {".", ".."}:
        return True
    if name.startswith(".tmp-") or name.startswith("."):
        return True
    return name.endswith(".tmp") or name.endswith(".partial")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("utf-8"),
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def format_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_dt(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_date(value: str | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value)[:10])


_UNSAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._:=+-]+")


def safe_segment(value: str) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    text = _UNSAFE_SEGMENT.sub("_", text)
    text = text.strip("._") or "_"
    if text in {".", ".."} or ".." in text:
        raise VisionError(f"unsafe path segment {value!r}")
    return text


def safe_join(root: Path, relative: str) -> Path:
    rel = str(relative).replace("\\", "/").lstrip("/")
    if not rel or rel.startswith("/") or Path(rel).is_absolute():
        raise VisionError(f"artifact path must be relative: {relative!r}")
    parts = Path(rel).parts
    if any(p in {"", ".", ".."} or p.startswith("/") for p in parts):
        raise VisionError(f"artifact path escapes run root: {relative!r}")
    root_resolved = root.resolve()
    full = (root_resolved / rel).resolve()
    try:
        full.relative_to(root_resolved)
    except ValueError as exc:
        raise VisionError(f"artifact path escapes run root: {relative!r}") from exc
    return full


def iter_json_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix == ".json" and not is_temp_name(path.name):
            yield path


# --- codecs -----------------------------------------------------------------


def encode_feature(feat: FeatureValue) -> dict[str, Any]:
    return {
        "close": feat.close,
        "dollar_vol_avg_20": feat.dollar_vol_avg_20,
        "adr_pct_20": feat.adr_pct_20,
    }


def decode_feature(payload: Mapping[str, Any]) -> FeatureValue:
    return FeatureValue(
        close=payload.get("close"),
        dollar_vol_avg_20=payload.get("dollar_vol_avg_20"),
        adr_pct_20=payload.get("adr_pct_20"),
    )


def encode_bar(bar: Bar) -> dict[str, Any]:
    return {
        "date": bar.date.isoformat(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def decode_bar(payload: Mapping[str, Any]) -> Bar:
    return Bar(
        date=parse_date(payload["date"]),
        open=payload.get("open"),
        high=payload.get("high"),
        low=payload.get("low"),
        close=payload.get("close"),
        volume=payload.get("volume"),
    )


def encode_window(window: BarWindow) -> dict[str, Any]:
    return {
        "ticker": window.ticker,
        "timeframe": window.timeframe,
        "bars": [encode_bar(b) for b in window.bars],
        "provenance": window.provenance,
        "source_digest": window.source_digest,
    }


def decode_window(payload: Mapping[str, Any]) -> BarWindow:
    return BarWindow(
        ticker=str(payload["ticker"]),
        timeframe=str(payload.get("timeframe") or "1d"),
        bars=tuple(decode_bar(b) for b in payload.get("bars") or ()),
        provenance=str(payload.get("provenance") or "unknown"),
        source_digest=str(payload["source_digest"]),
    )


def encode_profile(profile: ChartProfile) -> dict[str, Any]:
    return {
        "timeframe": profile.timeframe,
        "lookback_bars": profile.lookback_bars,
        "volume": profile.volume,
        "moving_averages": list(profile.moving_averages),
    }


def decode_profile(payload: Mapping[str, Any]) -> ChartProfile:
    return ChartProfile(
        timeframe=str(payload.get("timeframe") or "1d"),
        lookback_bars=int(payload["lookback_bars"]),
        volume=bool(payload.get("volume")),
        moving_averages=tuple(int(x) for x in payload.get("moving_averages") or ()),
    )


def encode_config(config: ScanConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "batch_size": config.batch_size,
        "max_concurrency": config.max_concurrency,
        "max_retries": config.max_retries,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "retry_backoff_max_seconds": config.retry_backoff_max_seconds,
        "jitter_ratio": config.jitter_ratio,
        "max_request_bytes": config.max_request_bytes,
        "max_images_per_request": config.max_images_per_request,
        "max_in_flight_batches": config.max_in_flight_batches,
        "image_detail": config.image_detail,
        "mode": config.mode,
        "request_label": config.request_label,
    }


def decode_config(payload: Mapping[str, Any]) -> ScanConfig:
    return ScanConfig(
        model=str(payload["model"]),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        max_output_tokens=int(payload.get("max_output_tokens", 4096)),
        batch_size=int(payload.get("batch_size", 10)),
        max_concurrency=int(payload.get("max_concurrency", 2)),
        max_retries=int(payload.get("max_retries", 3)),
        retry_backoff_seconds=float(payload.get("retry_backoff_seconds", 1.0)),
        retry_backoff_max_seconds=float(payload.get("retry_backoff_max_seconds", 20.0)),
        jitter_ratio=float(payload.get("jitter_ratio", 0.25)),
        max_request_bytes=int(payload.get("max_request_bytes", 2_500_000)),
        max_images_per_request=int(payload.get("max_images_per_request", 24)),
        max_in_flight_batches=int(payload.get("max_in_flight_batches", 2)),
        image_detail=payload.get("image_detail") or "auto",
        mode=payload.get("mode") or "live",
        request_label=payload.get("request_label"),
    )


def encode_diagnostics(diag: ScanDiagnostics) -> dict[str, Any]:
    counts = diag.per_setup_eligible_counts
    return {
        "universe_count": diag.universe_count,
        "input_count": diag.input_count,
        "eligible_union_count": diag.eligible_union_count,
        "per_setup_eligible_counts": [list(item) for item in counts] if counts is not None else None,
        "not_eligible_count": diag.not_eligible_count,
        "skipped_count": diag.skipped_count,
        "unavailable": list(diag.unavailable),
    }


def decode_diagnostics(payload: Mapping[str, Any]) -> ScanDiagnostics:
    raw = payload.get("per_setup_eligible_counts")
    per_setup = None
    if raw is not None:
        per_setup = tuple((str(k), int(v)) for k, v in raw)
    return ScanDiagnostics(
        universe_count=payload.get("universe_count"),
        input_count=payload.get("input_count"),
        eligible_union_count=payload.get("eligible_union_count"),
        per_setup_eligible_counts=per_setup,
        not_eligible_count=payload.get("not_eligible_count"),
        skipped_count=payload.get("skipped_count"),
        unavailable=tuple(payload.get("unavailable") or ()),
    )


def encode_assessment(item: Assessment) -> dict[str, Any]:
    return {
        "setup_id": item.setup_id,
        "verdict": item.verdict,
        "match_strength": item.match_strength,
        "reason": item.reason,
        "violated_required_rule_ids": list(item.violated_required_rule_ids),
        "missing_evidence": list(item.missing_evidence),
    }


def decode_assessment(payload: Mapping[str, Any]) -> Assessment:
    return Assessment(
        setup_id=str(payload["setup_id"]),
        verdict=payload["verdict"],
        match_strength=payload.get("match_strength"),
        reason=str(payload.get("reason") or ""),
        violated_required_rule_ids=tuple(payload.get("violated_required_rule_ids") or ()),
        missing_evidence=tuple(payload.get("missing_evidence") or ()),
    )


def encode_error(error: AttemptError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "kind": error.kind,
        "message": error.message,
        "retryable": error.retryable,
        "http_status": error.http_status,
        "provider_code": error.provider_code,
        "retry_after_seconds": error.retry_after_seconds,
    }


def decode_error(payload: Mapping[str, Any] | None) -> AttemptError | None:
    if not payload:
        return None
    return AttemptError(
        kind=payload["kind"],
        message=str(payload.get("message") or ""),
        retryable=bool(payload.get("retryable")),
        http_status=payload.get("http_status"),
        provider_code=payload.get("provider_code"),
        retry_after_seconds=payload.get("retry_after_seconds"),
    )


def encode_usage(usage: TokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def decode_usage(payload: Mapping[str, Any] | None) -> TokenUsage | None:
    if not payload:
        return None
    return TokenUsage(
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        total_tokens=payload.get("total_tokens"),
    )


def encode_ma(item: MaAvailability) -> dict[str, Any]:
    return {
        "period": item.period,
        "available": item.available,
        "first_valid_index": item.first_valid_index,
        "note": item.note,
    }


def decode_ma(payload: Mapping[str, Any]) -> MaAvailability:
    return MaAvailability(
        period=int(payload["period"]),
        available=bool(payload["available"]),
        first_valid_index=payload.get("first_valid_index"),
        note=str(payload.get("note") or ""),
    )


def encode_candidate_result(row: CandidateResult) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "status": row.status,
        "ticker": row.ticker,
        "asof_date": row.asof_date.isoformat(),
        "features": encode_feature(row.features),
        "eligible_setup_ids": list(row.eligible_setup_ids),
        "assessments": [encode_assessment(a) for a in row.assessments],
        "artifact_id": row.artifact_id,
        "error": encode_error(row.error),
        "attempt_id": row.attempt_id,
        "source_digest": row.source_digest,
    }


def decode_candidate_result(payload: Mapping[str, Any]) -> CandidateResult:
    return CandidateResult(
        candidate_id=str(payload["candidate_id"]),
        status=payload["status"],
        ticker=str(payload["ticker"]),
        asof_date=parse_date(payload["asof_date"]),
        features=decode_feature(payload.get("features") or {}),
        eligible_setup_ids=tuple(payload.get("eligible_setup_ids") or ()),
        assessments=tuple(decode_assessment(a) for a in payload.get("assessments") or ()),
        artifact_id=payload.get("artifact_id"),
        error=decode_error(payload.get("error")),
        attempt_id=payload.get("attempt_id"),
        source_digest=str(payload.get("source_digest") or ""),
    )


def result_content_digest(row: CandidateResult) -> str:
    return digest(encode_candidate_result(row))


def encode_attempt(attempt: ClassificationAttempt) -> dict[str, Any]:
    results = None
    if attempt.results is not None:
        results = [
            {
                "candidate_id": row.candidate_id,
                "assessments": [encode_assessment(a) for a in row.assessments],
            }
            for row in attempt.results
        ]
    return {
        "attempt_id": attempt.attempt_id,
        "batch_id": attempt.batch_id,
        "batch_fingerprint": attempt.batch_fingerprint,
        "candidate_ids": list(attempt.candidate_ids),
        "setup_ids": list(attempt.setup_ids),
        "started_at": format_dt(attempt.started_at),
        "ended_at": format_dt(attempt.ended_at) if attempt.ended_at else None,
        "provider": attempt.provider,
        "model": attempt.model,
        "response_id": attempt.response_id,
        "usage": encode_usage(attempt.usage),
        "retry_after_seconds": attempt.retry_after_seconds,
        "latency_ms": attempt.latency_ms,
        "error": encode_error(attempt.error),
        "sanitized_output": attempt.sanitized_output,
        "results": results,
        "accepted": attempt.accepted,
    }


def decode_attempt(payload: Mapping[str, Any]) -> ClassificationAttempt:
    raw_results = payload.get("results")
    results = None
    if raw_results is not None:
        results = tuple(
            CandidateClassification(
                candidate_id=str(row["candidate_id"]),
                assessments=tuple(decode_assessment(a) for a in row.get("assessments") or ()),
            )
            for row in raw_results
        )
    ended = payload.get("ended_at")
    return ClassificationAttempt(
        attempt_id=str(payload["attempt_id"]),
        batch_id=str(payload["batch_id"]),
        batch_fingerprint=str(payload["batch_fingerprint"]),
        candidate_ids=tuple(payload.get("candidate_ids") or ()),
        setup_ids=tuple(payload.get("setup_ids") or ()),
        started_at=parse_dt(payload["started_at"]),
        ended_at=parse_dt(ended) if ended else None,
        provider=str(payload.get("provider") or ""),
        model=payload.get("model"),
        response_id=payload.get("response_id"),
        usage=decode_usage(payload.get("usage")),
        retry_after_seconds=payload.get("retry_after_seconds"),
        latency_ms=payload.get("latency_ms"),
        error=decode_error(payload.get("error")),
        sanitized_output=payload.get("sanitized_output"),
        results=results,
        accepted=bool(payload.get("accepted")),
    )


def aggregate_attempt_usage(attempts: Sequence[ClassificationAttempt]) -> tuple[int, TokenUsage | None]:
    """Count each provider attempt_id once, even if raw and validated forms were journaled."""

    chosen: dict[str, ClassificationAttempt] = {}
    order: list[str] = []
    for attempt in attempts:
        aid = attempt.attempt_id
        if aid not in chosen:
            chosen[aid] = attempt
            order.append(aid)
            continue
        prev = chosen[aid]
        if attempt.accepted and not prev.accepted:
            chosen[aid] = attempt
        elif prev.usage is None and attempt.usage is not None:
            chosen[aid] = attempt
    known = [chosen[aid].usage for aid in order if chosen[aid].usage is not None]
    if not known:
        return len(chosen), None
    usage = TokenUsage(
        input_tokens=sum(u.input_tokens or 0 for u in known) or None,
        output_tokens=sum(u.output_tokens or 0 for u in known) or None,
        total_tokens=sum(u.total_tokens or 0 for u in known) or None,
    )
    return len(chosen), usage


def encode_compiler(snap: CompilerSnapshot | None) -> dict[str, Any] | None:
    if snap is None:
        return None
    return {
        "compiler_id": snap.compiler_id,
        "instructions": snap.instructions,
        "schema_name": snap.schema_name,
        "json_schema": dict(snap.json_schema),
        "fingerprint": snap.fingerprint,
    }


def decode_compiler(payload: Mapping[str, Any] | None) -> CompilerSnapshot | None:
    if not payload:
        return None
    return CompilerSnapshot(
        compiler_id=str(payload.get("compiler_id") or "snapshot_request_compiler_v1"),
        instructions=str(payload.get("instructions") or ""),
        schema_name=str(payload.get("schema_name") or ""),
        json_schema=dict(payload.get("json_schema") or {}),
        fingerprint=str(payload.get("fingerprint") or ""),
    )


def encode_skip(skip: InputSkip) -> dict[str, Any]:
    return {
        "ticker": skip.ticker,
        "kind": skip.kind,
        "message": skip.message,
        "asof_date": skip.asof_date.isoformat() if skip.asof_date else None,
        "bar_count": skip.bar_count,
        "features": None if skip.features is None else encode_feature(skip.features),
    }


def decode_skip(payload: Mapping[str, Any]) -> InputSkip:
    raw_feat = payload.get("features")
    return InputSkip(
        ticker=str(payload["ticker"]),
        kind=payload["kind"],
        message=str(payload.get("message") or ""),
        asof_date=parse_date(payload["asof_date"]) if payload.get("asof_date") else None,
        bar_count=payload.get("bar_count"),
        features=decode_feature(raw_feat) if isinstance(raw_feat, Mapping) else None,
    )


def _result_write_action(prior: CandidateResult | None, row: CandidateResult) -> str:
    if prior is None:
        return "write"
    if prior.status == "error":
        return "write"
    if result_content_digest(prior) == result_content_digest(row):
        return "skip"
    return "conflict"


def encode_artifact_index_entry(ref: ArtifactRef, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "artifact_id": ref.artifact_id,
        "relative_path": ref.relative_path,
        "sha256": ref.sha256,
        "width": ref.width,
        "height": ref.height,
        "byte_length": ref.byte_length,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def decode_artifact_ref(payload: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(payload["artifact_id"]),
        relative_path=str(payload["relative_path"]),
        sha256=str(payload["sha256"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        byte_length=int(payload["byte_length"]),
    )


def new_run_id(now: datetime) -> str:
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{RUN_DIR_PREFIX}{stamp}-{uuid4().hex[:10]}"


class _RunProcessLock:
    """Advisory exclusive lock released when the process dies (fcntl.flock)."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._fh: Any = None

    def __enter__(self) -> "_RunProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows fallback
                import msvcrt

                self._fh.seek(0)
                if self._fh.read(1) == b"":
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise RunLockedError(f"run {self.run_id} is locked") from exc
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None


class FilesystemRunStore:
    """Durable vision-scan store. Inject ``scan_root``; Agent 3 chooses the project path."""

    def __init__(self, scan_root: Path, *, clock: Clock | None = None) -> None:
        self.scan_root = Path(scan_root)
        self.scan_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or utcnow
        self._thread_lock = Lock()

    def run_dir(self, run_id: str) -> Path:
        rid = safe_segment(run_id)
        if not rid.startswith(RUN_DIR_PREFIX):
            raise VisionError(f"invalid run_id {run_id!r}")
        path = (self.scan_root / rid).resolve()
        try:
            path.relative_to(self.scan_root.resolve())
        except ValueError as exc:
            raise VisionError(f"run_id escapes scan root: {run_id!r}") from exc
        return path

    def create_run(self, prepared: PreparedScan) -> StoredRun:
        now = self._clock()
        run_id = new_run_id(now)
        directory = self.scan_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        status = "dry_run" if prepared.config.mode == "dry_run" else "running"
        stored = StoredRun(
            run_id=run_id,
            status=status,
            created_at=now,
            updated_at=now,
            semantic_digest=prepared.semantic_digest,
            raw_digest=prepared.raw_digest,
            config=prepared.config,
            profile=prepared.profile,
            lock_holder=None,
            synthetic=prepared.config.mode == "demo",
        )
        self._write_frozen(directory, prepared)
        self._write_meta(directory, stored, summary=None)
        atomic_write_json(directory / "artifacts" / ARTIFACT_INDEX_NAME, {"artifacts": []})
        self._rebuild_derived(run_id)
        return stored

    def lock_run(self, run_id: str):
        directory = self.run_dir(run_id)
        if not directory.exists():
            raise VisionError(f"unknown run {run_id}")
        return _RunProcessLock(directory / LOCK_NAME, run_id)

    def load_run(self, run_id: str) -> StoredRun:
        return self._load_meta(self.run_dir(run_id)).stored

    def load_frozen_inputs(self, run_id: str) -> PreparedScan:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        return self._read_frozen(directory)

    def save_artifact(self, run_id: str, artifact: ChartArtifact) -> ArtifactRef:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        rel = self._artifact_relative_path(artifact)
        dest = safe_join(directory, rel)
        png = artifact.image.png_bytes
        digest = artifact.image.sha256 or digest_bytes(png)
        ref = ArtifactRef(
            artifact_id=artifact.artifact_id,
            relative_path=rel,
            sha256=digest,
            width=int(artifact.image.width),
            height=int(artifact.image.height),
            byte_length=len(png),
        )
        extra = {
            "kind": artifact.kind,
            "title": artifact.title,
            "candidate_id": artifact.candidate_id,
            "setup_id": artifact.setup_id,
            "example_id": artifact.example_id,
            "source": artifact.source,
            "final_session_date": artifact.final_session_date.isoformat() if artifact.final_session_date else None,
            "profile": encode_profile(artifact.profile) if artifact.profile else None,
            "ma_availability": [encode_ma(m) for m in artifact.ma_availability],
        }
        with self._thread_lock:
            index = self._read_artifact_index(directory)
            same_path = next((e for e in index if e.get("relative_path") == rel), None)
            if same_path is not None:
                if same_path.get("sha256") != ref.sha256 or int(same_path.get("byte_length") or 0) != ref.byte_length:
                    raise VisionError(f"artifact path {rel} already used with different bytes")
                return decode_artifact_ref(same_path)
            if dest.exists():
                on_disk = dest.read_bytes()
                if digest_bytes(on_disk) != ref.sha256:
                    raise VisionError(f"artifact path {rel} already used with different bytes")
            else:
                atomic_write_bytes(dest, png)
            index.append(encode_artifact_index_entry(ref, extra))
            self._write_artifact_index(directory, index)
        return ref

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes:
        ref = self.get_artifact_ref(run_id, artifact_id)
        path = safe_join(self.run_dir(run_id), ref.relative_path)
        if not path.is_file():
            raise VisionError(f"saved chart missing for artifact {artifact_id}")
        data = path.read_bytes()
        if digest_bytes(data) != ref.sha256:
            raise VisionError(f"saved chart digest mismatch for artifact {artifact_id}")
        return data

    def get_artifact_ref(self, run_id: str, artifact_id: str) -> ArtifactRef:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        entries = [e for e in self._read_artifact_index(directory) if e.get("artifact_id") == artifact_id]
        if not entries:
            raise VisionError(f"unknown artifact {artifact_id} in run {run_id}")
        entries.sort(key=lambda e: 0 if e.get("kind") == "candidate" else 1)
        return decode_artifact_ref(entries[0])

    def find_artifact_for_candidate(self, run_id: str, candidate_id: str) -> ArtifactRef | None:
        directory = self.run_dir(run_id)
        if not directory.exists():
            return None
        for entry in self._read_artifact_index(directory):
            if entry.get("candidate_id") == candidate_id:
                return decode_artifact_ref(entry)
        return None

    def artifact_entry(self, run_id: str, artifact_id: str) -> dict[str, Any] | None:
        directory = self.run_dir(run_id)
        for entry in self._read_artifact_index(directory):
            if entry.get("artifact_id") == artifact_id:
                return dict(entry)
        return None

    def journal_attempt(self, run_id: str, attempt: ClassificationAttempt) -> None:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        payload = encode_attempt(attempt)
        payload_digest = digest(payload)
        attempts_dir = directory / "attempts"
        for path in iter_json_files(attempts_dir):
            if digest(read_json(path)) == payload_digest:
                return
        name = f"{safe_segment(attempt.attempt_id)}-{payload_digest.split(':', 1)[-1][:12]}.json"
        atomic_write_json(attempts_dir / name, payload)

    def recover_attempt(self, run_id: str, batch_fingerprint: str) -> ClassificationAttempt | None:
        usable: list[ClassificationAttempt] = []
        for attempt in self.list_attempts(run_id):
            if attempt.batch_fingerprint != batch_fingerprint:
                continue
            if attempt.error is not None:
                continue
            if attempt.results is None and attempt.sanitized_output is None:
                continue
            usable.append(attempt)
        if not usable:
            return None
        usable.sort(key=lambda a: (bool(a.accepted), a.ended_at or a.started_at, a.attempt_id))
        return usable[-1]

    def commit_batch(
        self,
        run_id: str,
        *,
        batch_id: str,
        results: Sequence[CandidateResult],
        attempt: ClassificationAttempt,
    ) -> None:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        rows = tuple(results)
        shard = {
            "batch_id": batch_id,
            "attempt_id": attempt.attempt_id,
            "candidate_ids": [r.candidate_id for r in rows],
            "results": [encode_candidate_result(r) for r in rows],
            "content_digest": digest([encode_candidate_result(r) for r in rows]),
        }
        batch_path = directory / "batches" / f"{safe_segment(batch_id)}.json"
        with self._thread_lock:
            if batch_path.exists():
                existing = read_json(batch_path)
                if digest(existing.get("results")) != digest(shard["results"]):
                    raise VisionError(f"batch {batch_id} already committed with different content")
                self.journal_attempt(run_id, attempt)
                return
            for row in rows:
                prior = self._read_result(directory, row.candidate_id)
                action = _result_write_action(prior, row)
                if action == "conflict":
                    raise VisionError(
                        f"candidate {row.candidate_id} already committed with different content"
                    )
            self.journal_attempt(run_id, attempt)
            atomic_write_json(batch_path, shard)
            for row in rows:
                self._write_result(directory, row, source="commit", batch_id=batch_id)
        self._rebuild_derived(run_id)

    def mark_candidates(self, run_id: str, results: Sequence[CandidateResult]) -> None:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        with self._thread_lock:
            for row in results:
                prior = self._read_result(directory, row.candidate_id)
                action = _result_write_action(prior, row)
                if action == "conflict":
                    raise VisionError(
                        f"candidate {row.candidate_id} already recorded with different content"
                    )
                if action == "skip":
                    continue
                self._write_result(directory, row, source="mark", batch_id=None)
        self._rebuild_derived(run_id)

    def list_committed_candidate_ids(self, run_id: str) -> frozenset[str]:
        return frozenset(r.candidate_id for r in self.list_candidate_results(run_id) if r.status == "completed")

    def list_candidate_results(self, run_id: str) -> tuple[CandidateResult, ...]:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        self._recover_results_from_batches(directory)
        rows = [decode_candidate_result(read_json(path)) for path in iter_json_files(directory / "results")]
        rows.sort(key=lambda r: r.candidate_id)
        return tuple(rows)

    def list_attempts(self, run_id: str) -> tuple[ClassificationAttempt, ...]:
        directory = self.run_dir(run_id)
        self._require_schema(directory)
        attempts = [decode_attempt(read_json(path)) for path in iter_json_files(directory / "attempts")]
        attempts.sort(key=lambda a: (a.started_at, a.attempt_id))
        return tuple(attempts)

    def update_status(self, run_id: str, status: str) -> StoredRun:
        directory = self.run_dir(run_id)
        bundle = self._load_meta(directory)
        stored = StoredRun(
            run_id=bundle.stored.run_id,
            status=status,  # type: ignore[arg-type]
            created_at=bundle.stored.created_at,
            updated_at=self._clock(),
            semantic_digest=bundle.stored.semantic_digest,
            raw_digest=bundle.stored.raw_digest,
            config=bundle.stored.config,
            profile=bundle.stored.profile,
            lock_holder=bundle.stored.lock_holder,
            synthetic=bundle.stored.synthetic,
        )
        self._write_meta(directory, stored, summary=bundle.summary)
        return stored

    def finalize(self, run_id: str, summary: RunSummary) -> StoredRun:
        directory = self.run_dir(run_id)
        bundle = self._load_meta(directory)
        stored = StoredRun(
            run_id=bundle.stored.run_id,
            status=summary.status,
            created_at=bundle.stored.created_at,
            updated_at=self._clock(),
            semantic_digest=bundle.stored.semantic_digest,
            raw_digest=bundle.stored.raw_digest,
            config=bundle.stored.config,
            profile=bundle.stored.profile,
            lock_holder=None,
            synthetic=summary.synthetic or bundle.stored.synthetic,
        )
        self._write_meta(directory, stored, summary=summary)
        self._rebuild_derived(run_id)
        return stored

    def export_manifest(self, run_id: str) -> dict[str, Any]:
        directory = self.run_dir(run_id)
        manifest_path = directory / "derived" / "manifest.json"
        if not manifest_path.exists():
            self._rebuild_derived(run_id)
        return read_json(manifest_path)

    def list_runs(self) -> tuple[StoredRun, ...]:
        found: list[StoredRun] = []
        if not self.scan_root.is_dir():
            return ()
        for path in self.scan_root.iterdir():
            if not path.is_dir() or is_temp_name(path.name) or not path.name.startswith(RUN_DIR_PREFIX):
                continue
            meta = path / META_NAME
            if not meta.is_file():
                continue
            try:
                found.append(self._load_meta(path).stored)
            except VisionError:
                continue
        found.sort(key=lambda r: (r.created_at, r.run_id), reverse=True)
        return tuple(found)

    # --- internals ----------------------------------------------------------

    def _artifact_relative_path(self, artifact: ChartArtifact) -> str:
        if artifact.kind == "candidate":
            name = artifact.candidate_id or artifact.artifact_id
            return f"artifacts/candidates/{safe_segment(name)}.png"
        setup_id = artifact.setup_id or "_"
        example_id = artifact.example_id or artifact.artifact_id
        return f"artifacts/examples/{safe_segment(setup_id)}/{safe_segment(example_id)}.png"

    def _write_frozen(self, directory: Path, prepared: PreparedScan) -> None:
        frozen_dir = directory / "frozen"
        yaml_dir = frozen_dir / "yaml"
        windows_dir = frozen_dir / "windows"
        example_bytes_dir = frozen_dir / "example_bytes"
        setups_payload = []
        for snap in prepared.setups:
            rel = f"frozen/yaml/{safe_segment(snap.setup_id)}.yaml"
            atomic_write_bytes(directory / rel, snap.yaml_bytes)
            setups_payload.append(
                {
                    "setup_id": snap.setup_id,
                    "name": snap.name,
                    "enabled": snap.enabled,
                    "timeframe": snap.timeframe,
                    "lookback_bars": snap.lookback_bars,
                    "description": snap.description,
                    "llm_notes": snap.llm_notes,
                    "rules": [
                        {
                            "rule_id": rule.rule_id,
                            "kind": rule.kind,
                            "index": rule.index,
                            "text": rule.text,
                        }
                        for rule in snap.rules
                    ],
                    "filters": {
                        "min_price": snap.filters.min_price,
                        "min_dollar_vol_20d": snap.filters.min_dollar_vol_20d,
                        "min_adr_pct_20": snap.filters.min_adr_pct_20,
                        "min_market_cap": snap.filters.min_market_cap,
                        "max_market_cap": snap.filters.max_market_cap,
                    },
                    "chart_volume": snap.chart_volume,
                    "chart_moving_averages": list(snap.chart_moving_averages),
                    "yaml_rel": rel,
                    "yaml_digest": snap.yaml_digest,
                    "content_digest": snap.content_digest,
                    "example_ids": list(snap.example_ids),
                }
            )
        examples_payload = []
        for ex in prepared.examples:
            image_rel = None
            if ex.image_bytes is not None:
                image_rel = (
                    f"frozen/example_bytes/{safe_segment(ex.setup_id)}/{safe_segment(ex.example_id)}.bin"
                )
                atomic_write_bytes(directory / image_rel, ex.image_bytes)
            window_rel = None
            if ex.window is not None:
                window_rel = (
                    f"frozen/windows/examples/{safe_segment(ex.setup_id)}__{safe_segment(ex.example_id)}.json"
                )
                atomic_write_json(directory / window_rel, encode_window(ex.window))
            examples_payload.append(
                {
                    "setup_id": ex.setup_id,
                    "example_id": ex.example_id,
                    "polarity": ex.polarity,
                    "quality": ex.quality,
                    "note": ex.note,
                    "type": ex.type,
                    "ticker": ex.ticker,
                    "asof_date": ex.asof_date.isoformat() if ex.asof_date else None,
                    "timeframe": ex.timeframe,
                    "image_rel": image_rel,
                    "window_rel": window_rel,
                    "raw_digest": ex.raw_digest,
                }
            )
        candidates_payload = []
        for cand in prepared.candidates:
            window_rel = f"frozen/windows/candidates/{safe_segment(cand.candidate_id)}.json"
            atomic_write_json(directory / window_rel, encode_window(cand.window))
            candidates_payload.append(
                {
                    "candidate_id": cand.candidate_id,
                    "ticker": cand.ticker,
                    "window_rel": window_rel,
                    "features": encode_feature(cand.features),
                    "eligible_setup_ids": list(cand.eligible_setup_ids),
                    "source_digest": cand.source_digest,
                    "asof_date": cand.asof_date.isoformat(),
                    "profile": encode_profile(cand.profile),
                }
            )
        gf_rel = "frozen/global_filters.yaml"
        atomic_write_bytes(directory / gf_rel, prepared.global_filters_yaml_bytes)
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "profile": encode_profile(prepared.profile),
            "setups": setups_payload,
            "examples": examples_payload,
            "candidates": candidates_payload,
            "skips": [encode_skip(s) for s in prepared.skips],
            "config": encode_config(prepared.config),
            "diagnostics": encode_diagnostics(prepared.diagnostics),
            "prepared_at": format_dt(prepared.prepared_at),
            "semantic_digest": prepared.semantic_digest,
            "raw_digest": prepared.raw_digest,
            "global_filters_rel": gf_rel,
            "global_filters_raw_digest": prepared.global_filters_raw_digest,
            "compiler": encode_compiler(prepared.compiler),
        }
        yaml_dir.mkdir(parents=True, exist_ok=True)
        windows_dir.mkdir(parents=True, exist_ok=True)
        example_bytes_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / FROZEN_NAME, envelope)

    def _read_frozen(self, directory: Path) -> PreparedScan:
        envelope = read_json(directory / FROZEN_NAME)
        setups = []
        for raw in envelope.get("setups") or ():
            yaml_bytes = safe_join(directory, raw["yaml_rel"]).read_bytes()
            filters = raw.get("filters") or {}
            setups.append(
                SetupSnapshot(
                    setup_id=str(raw["setup_id"]),
                    name=str(raw["name"]),
                    enabled=bool(raw.get("enabled", True)),
                    timeframe=str(raw.get("timeframe") or "1d"),
                    lookback_bars=int(raw["lookback_bars"]),
                    description=str(raw.get("description") or ""),
                    llm_notes=str(raw.get("llm_notes") or ""),
                    rules=tuple(
                        SnapshotRule(
                            rule_id=str(rule["rule_id"]),
                            kind=rule["kind"],
                            index=int(rule["index"]),
                            text=str(rule["text"]),
                        )
                        for rule in raw.get("rules") or ()
                    ),
                    filters=FilterSnapshot(
                        min_price=filters.get("min_price"),
                        min_dollar_vol_20d=filters.get("min_dollar_vol_20d"),
                        min_adr_pct_20=filters.get("min_adr_pct_20"),
                        min_market_cap=filters.get("min_market_cap"),
                        max_market_cap=filters.get("max_market_cap"),
                    ),
                    chart_volume=bool(raw.get("chart_volume")),
                    chart_moving_averages=tuple(int(x) for x in raw.get("chart_moving_averages") or ()),
                    yaml_bytes=yaml_bytes,
                    yaml_digest=str(raw["yaml_digest"]),
                    content_digest=str(raw["content_digest"]),
                    example_ids=tuple(raw.get("example_ids") or ()),
                )
            )
        examples = []
        for raw in envelope.get("examples") or ():
            image_bytes = None
            if raw.get("image_rel"):
                image_bytes = safe_join(directory, raw["image_rel"]).read_bytes()
            window = None
            if raw.get("window_rel"):
                window = decode_window(read_json(safe_join(directory, raw["window_rel"])))
            asof = raw.get("asof_date")
            examples.append(
                ExampleInput(
                    setup_id=str(raw["setup_id"]),
                    example_id=str(raw["example_id"]),
                    polarity=raw["polarity"],
                    quality=raw.get("quality"),
                    note=str(raw.get("note") or ""),
                    type=raw["type"],
                    ticker=raw.get("ticker"),
                    asof_date=parse_date(asof) if asof else None,
                    timeframe=str(raw.get("timeframe") or "1d"),
                    image_bytes=image_bytes,
                    window=window,
                    raw_digest=str(raw["raw_digest"]),
                )
            )
        candidates = []
        for raw in envelope.get("candidates") or ():
            window = decode_window(read_json(safe_join(directory, raw["window_rel"])))
            candidates.append(
                CandidateInput(
                    candidate_id=str(raw["candidate_id"]),
                    ticker=str(raw["ticker"]),
                    window=window,
                    features=decode_feature(raw.get("features") or {}),
                    eligible_setup_ids=tuple(raw.get("eligible_setup_ids") or ()),
                    source_digest=str(raw["source_digest"]),
                    asof_date=parse_date(raw["asof_date"]),
                    profile=decode_profile(raw["profile"]),
                )
            )
        skips = tuple(decode_skip(s) for s in envelope.get("skips") or ())
        gf_bytes = safe_join(directory, envelope["global_filters_rel"]).read_bytes()
        return PreparedScan(
            profile=decode_profile(envelope["profile"]),
            setups=tuple(setups),
            examples=tuple(examples),
            candidates=tuple(candidates),
            skips=skips,
            config=decode_config(envelope["config"]),
            diagnostics=decode_diagnostics(envelope.get("diagnostics") or {}),
            prepared_at=parse_dt(envelope["prepared_at"]),
            semantic_digest=str(envelope["semantic_digest"]),
            raw_digest=str(envelope["raw_digest"]),
            global_filters_yaml_bytes=gf_bytes,
            global_filters_raw_digest=str(envelope["global_filters_raw_digest"]),
            compiler=decode_compiler(envelope.get("compiler")),
        )

    def _write_meta(self, directory: Path, stored: StoredRun, summary: RunSummary | None) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": stored.run_id,
            "status": stored.status,
            "created_at": format_dt(stored.created_at),
            "updated_at": format_dt(stored.updated_at),
            "semantic_digest": stored.semantic_digest,
            "raw_digest": stored.raw_digest,
            "config": encode_config(stored.config),
            "profile": encode_profile(stored.profile),
            "lock_holder": stored.lock_holder,
            "synthetic": stored.synthetic,
            "summary": None
            if summary is None
            else {
                "candidates_total": summary.candidates_total,
                "candidates_completed": summary.candidates_completed,
                "candidates_error": summary.candidates_error,
                "candidates_skipped": summary.candidates_skipped,
                "candidates_pending": summary.candidates_pending,
                "setup_matches": summary.setup_matches,
                "attempts": summary.attempts,
                "usage": encode_usage(summary.usage),
                "status": summary.status,
                "synthetic": summary.synthetic,
            },
        }
        atomic_write_json(directory / META_NAME, payload)

    def _load_meta(self, directory: Path) -> "_MetaBundle":
        if not directory.exists():
            raise VisionError(f"unknown run {directory.name}")
        meta_path = directory / META_NAME
        if not meta_path.is_file():
            raise VisionError(f"run {directory.name} is missing {META_NAME}")
        payload = read_json(meta_path)
        self._check_schema(payload.get("schema_version"), directory.name)
        summary_raw = payload.get("summary")
        summary = None
        if summary_raw:
            summary = RunSummary(
                candidates_total=int(summary_raw.get("candidates_total") or 0),
                candidates_completed=int(summary_raw.get("candidates_completed") or 0),
                candidates_error=int(summary_raw.get("candidates_error") or 0),
                candidates_skipped=int(summary_raw.get("candidates_skipped") or 0),
                candidates_pending=int(summary_raw.get("candidates_pending") or 0),
                setup_matches=int(summary_raw.get("setup_matches") or 0),
                attempts=int(summary_raw.get("attempts") or 0),
                usage=decode_usage(summary_raw.get("usage")),
                status=summary_raw.get("status") or payload["status"],
                synthetic=bool(summary_raw.get("synthetic")),
            )
        stored = StoredRun(
            run_id=str(payload["run_id"]),
            status=payload["status"],
            created_at=parse_dt(payload["created_at"]),
            updated_at=parse_dt(payload["updated_at"]),
            semantic_digest=str(payload["semantic_digest"]),
            raw_digest=str(payload["raw_digest"]),
            config=decode_config(payload["config"]),
            profile=decode_profile(payload["profile"]),
            lock_holder=payload.get("lock_holder"),
            synthetic=bool(payload.get("synthetic")),
        )
        return _MetaBundle(stored=stored, summary=summary)

    def _require_schema(self, directory: Path) -> None:
        meta_path = directory / META_NAME
        if not meta_path.is_file():
            raise VisionError(f"run {directory.name} is missing {META_NAME}")
        payload = read_json(meta_path)
        self._check_schema(payload.get("schema_version"), directory.name)

    def _check_schema(self, version: Any, run_id: str) -> None:
        try:
            ver = int(version)
        except (TypeError, ValueError):
            ver = -1
        if ver != SCHEMA_VERSION:
            raise VisionError(
                f"Unsupported vision scan schema version {version!r} for run {run_id}. "
                "Existing files were not modified."
            )

    def _read_artifact_index(self, directory: Path) -> list[dict[str, Any]]:
        path = directory / "artifacts" / ARTIFACT_INDEX_NAME
        if not path.is_file():
            return []
        payload = read_json(path)
        return list(payload.get("artifacts") or [])

    def _write_artifact_index(self, directory: Path, entries: Sequence[Mapping[str, Any]]) -> None:
        atomic_write_json(directory / "artifacts" / ARTIFACT_INDEX_NAME, {"artifacts": list(entries)})

    def _result_path(self, directory: Path, candidate_id: str) -> Path:
        return directory / "results" / f"{safe_segment(candidate_id)}.json"

    def _read_result(self, directory: Path, candidate_id: str) -> CandidateResult | None:
        path = self._result_path(directory, candidate_id)
        if not path.is_file():
            return None
        return decode_candidate_result(read_json(path))

    def _write_result(
        self,
        directory: Path,
        row: CandidateResult,
        *,
        source: str,
        batch_id: str | None,
    ) -> None:
        payload = encode_candidate_result(row)
        payload["source"] = source
        payload["batch_id"] = batch_id
        atomic_write_json(self._result_path(directory, row.candidate_id), payload)

    def _recover_results_from_batches(self, directory: Path) -> None:
        results_dir = directory / "results"
        for path in iter_json_files(directory / "batches"):
            shard = read_json(path)
            for raw in shard.get("results") or ():
                cid = str(raw["candidate_id"])
                dest = self._result_path(directory, cid)
                if dest.is_file():
                    continue
                payload = dict(raw)
                payload["source"] = "commit"
                payload["batch_id"] = shard.get("batch_id")
                atomic_write_json(dest, payload)
        results_dir.mkdir(parents=True, exist_ok=True)

    def _rebuild_derived(self, run_id: str) -> None:
        directory = self.run_dir(run_id)
        try:
            stored = self._load_meta(directory).stored
        except VisionError:
            return
        self._recover_results_from_batches(directory)
        rows = [decode_candidate_result(read_json(p)) for p in iter_json_files(directory / "results")]
        artifacts = self._read_artifact_index(directory)
        attempts = [decode_attempt(read_json(p)) for p in iter_json_files(directory / "attempts")]
        attempt_count, usage = aggregate_attempt_usage(attempts)
        manifest = {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "status": stored.status,
            "created_at": format_dt(stored.created_at),
            "updated_at": format_dt(stored.updated_at),
            "semantic_digest": stored.semantic_digest,
            "raw_digest": stored.raw_digest,
            "model": stored.config.model,
            "mode": stored.config.mode,
            "synthetic": stored.synthetic,
            "usage": encode_usage(usage),
            "attempts": attempt_count,
            "artifacts": [
                {
                    "artifact_id": e.get("artifact_id"),
                    "relative_path": e.get("relative_path"),
                    "sha256": e.get("sha256"),
                    "byte_length": e.get("byte_length"),
                }
                for e in artifacts
            ],
            "candidate_counts": {
                "recorded": len(rows),
                "completed": sum(1 for r in rows if r.status == "completed"),
                "error": sum(1 for r in rows if r.status == "error"),
                "skipped": sum(1 for r in rows if r.status == "skipped"),
                "pending": sum(1 for r in rows if r.status == "pending"),
                "setup_matches": sum(sum(1 for a in r.assessments if a.verdict == "match") for r in rows),
            },
        }
        atomic_write_json(directory / "derived" / "manifest.json", manifest)
        atomic_write_json(
            directory / "derived" / "counters.json",
            manifest["candidate_counts"],
        )
        self._write_convenience_parquet(directory, rows)

    def _write_convenience_parquet(self, directory: Path, rows: Sequence[CandidateResult]) -> None:
        try:
            import pandas as pd
        except ImportError:
            return
        cand_records = []
        assess_records = []
        for row in rows:
            matched = [a for a in row.assessments if a.verdict == "match"]
            cand_records.append(
                {
                    "candidate_id": row.candidate_id,
                    "ticker": row.ticker,
                    "asof_date": row.asof_date.isoformat(),
                    "status": row.status,
                    "close": row.features.close,
                    "dollar_vol_avg_20": row.features.dollar_vol_avg_20,
                    "adr_pct_20": row.features.adr_pct_20,
                    "matched_setup_ids": ";".join(a.setup_id for a in matched),
                    "eligible_setup_ids": ";".join(row.eligible_setup_ids),
                    "error_kind": row.error.kind if row.error else None,
                    "error_message": row.error.message if row.error else None,
                    "artifact_id": row.artifact_id,
                }
            )
            for a in row.assessments:
                assess_records.append(
                    {
                        "candidate_id": row.candidate_id,
                        "ticker": row.ticker,
                        "setup_id": a.setup_id,
                        "verdict": a.verdict,
                        "match_strength": a.match_strength,
                        "reason": a.reason,
                        "violated_required_rule_ids": ";".join(a.violated_required_rule_ids),
                        "missing_evidence": ";".join(a.missing_evidence),
                    }
                )
        cand_path = directory / "derived" / "candidates.parquet"
        assess_path = directory / "derived" / "assessments.parquet"
        pd.DataFrame(cand_records).to_parquet(cand_path.with_suffix(".parquet.tmp"), index=False)
        os.replace(cand_path.with_suffix(".parquet.tmp"), cand_path)
        pd.DataFrame(assess_records).to_parquet(assess_path.with_suffix(".parquet.tmp"), index=False)
        os.replace(assess_path.with_suffix(".parquet.tmp"), assess_path)


@dataclass(frozen=True)
class _MetaBundle:
    stored: StoredRun
    summary: RunSummary | None


def open_vision_scans(scan_root: Path, *, clock: Clock | None = None):
    """Return ``(store, reader, reviews)`` sharing ``scan_root``."""

    from .query import FilesystemResultReader
    from .reviews import FilesystemReviewStore

    store = FilesystemRunStore(scan_root, clock=clock)
    reviews = FilesystemReviewStore(scan_root, clock=clock)
    reader = FilesystemResultReader(scan_root, store=store, reviews=reviews)
    return store, reader, reviews
