"""Background vision-scan jobs for the Streamlit Results page.

The scanner already commits each batch to the filesystem store. This module
runs ``VisionApp.run`` off the Streamlit script thread so the page can poll
that store and show candidates as they land.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4
import logging
import os
import time

from screener_loader.config import LoaderConfig
from screener_loader.update import update_market_data
from screener_loader.update_state import ensure_fresh_market_data
from screener_loader.vision.types import ScanOutcome

if TYPE_CHECKING:
    from screener_loader.ui.scan_panel import ScanLaunchSpec

log = logging.getLogger("screener_loader.ui.scan_jobs")

JOB_SESSION_KEY = "vs_job_id"
FOLLOW_JOB_KEY = "vs_follow_job"
LIVE_REFRESH_SECONDS = 1.5

AppFactory = Callable[..., Any]

_JOBS: dict[str, "ScanJob"] = {}
_JOBS_LOCK = Lock()


@dataclass
class ScanProgress:
    phase: str
    run_id: str | None
    error: str | None
    done: int
    total: int
    completed: int
    pending: int
    skipped: int
    error_count: int
    matches: int
    fraction: float
    caption: str
    ticker: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None


def format_clock(seconds: float) -> str:
    """Compact ``m:ss`` or ``h:mm:ss`` for progress captions."""

    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_progress_caption(
    *,
    phase: str,
    error: str | None,
    ticker: str | None,
    done: int,
    total: int,
    elapsed_s: float,
    eta_s: float | None,
) -> str:
    """Ticker, settled count, elapsed time, and a remaining estimate. No run ids."""

    if error:
        return error
    if phase == "cancelling":
        parts = ["Cancelling"]
        if ticker:
            parts.append(ticker)
        if total > 0:
            parts.append(f"{int(done)}/{int(total)}")
        parts.append(f"{format_clock(elapsed_s)} elapsed")
        return " · ".join(parts)
    active = phase in {"preparing", "running"}
    parts = []
    if active and ticker:
        parts.append(ticker)
    if total > 0:
        parts.append(f"{int(done)}/{int(total)}")
    if active:
        if not parts:
            parts.append("Starting")
        parts.append(f"{format_clock(elapsed_s)} elapsed")
        if eta_s is not None and total > done:
            parts.append(f"~{format_clock(eta_s)} left")
    else:
        parts.insert(0, "Cancelled" if phase == "cancelled" else "Done")
        parts.append(format_clock(elapsed_s))
    return " · ".join(parts)


class ScanJob:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.cancel = Event()
        self._lock = Lock()
        self.phase = "preparing"
        self.run_id: str | None = None
        self.error: str | None = None
        self.outcome: ScanOutcome | None = None
        self.thread: Thread | None = None
        self.started_at = time.monotonic()
        self.ended_at: float | None = None
        self.current_ticker: str | None = None

    def is_active(self) -> bool:
        with self._lock:
            return self.phase in {"preparing", "running", "cancelling"}

    def request_cancel(self) -> None:
        """Ask the worker to stop, and show that immediately."""

        self.cancel.set()
        with self._lock:
            if self.phase in {"preparing", "running", "cancelling"}:
                self.phase = "cancelling"

    def mark_cancelled(self) -> None:
        with self._lock:
            self.phase = "cancelled"
            self.error = None
            self._mark_ended()

    def snapshot(self) -> tuple[str, str | None, str | None, ScanOutcome | None]:
        with self._lock:
            return self.phase, self.run_id, self.error, self.outcome

    def set_run_id(self, run_id: str) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            return
        with self._lock:
            self.run_id = rid
            if self.phase == "preparing":
                self.phase = "running"

    def set_current_ticker(self, ticker: str) -> None:
        text = str(ticker or "").strip().upper()
        if not text:
            return
        with self._lock:
            self.current_ticker = text

    def current_ticker_name(self) -> str | None:
        with self._lock:
            return self.current_ticker

    def elapsed_seconds(self) -> float:
        with self._lock:
            end = self.ended_at
            start = self.started_at
        if end is None:
            end = time.monotonic()
        return max(0.0, end - start)

    def _mark_ended(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.monotonic()

    def finish(self, outcome: ScanOutcome) -> None:
        with self._lock:
            self.outcome = outcome
            self.run_id = outcome.run_id
            self.phase = "cancelled" if self.cancel.is_set() else "done"
            self.error = None
            self._mark_ended()

    def fail(self, exc: BaseException) -> None:
        from screener_loader.ui.scan_panel import format_scan_error

        with self._lock:
            self.phase = "error"
            self.error = format_scan_error(exc)
            self._mark_ended()


def should_live_refresh(job: ScanJob | None) -> bool:
    """Auto-refresh the results fragment while a UI job is in flight.

    Pytest skips the timer so AppTest cannot loop on ``run_every``.
    """

    if job is None or not job.is_active():
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def get_job(job_id: str | None) -> ScanJob | None:
    if not job_id:
        return None
    with _JOBS_LOCK:
        return _JOBS.get(str(job_id))


def register_job(job: ScanJob) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def start_scan_job(
    cfg: LoaderConfig,
    spec: ScanLaunchSpec,
    *,
    resume_id: str | None = None,
    app_factory: AppFactory | None = None,
) -> ScanJob:
    from screener_loader.ui.scan_panel import launch_scan, resume_scan

    job = ScanJob(uuid4().hex)
    register_job(job)

    def worker() -> None:
        try:
            if job.cancel.is_set():
                job.mark_cancelled()
                return
            if resume_id:
                job.set_run_id(resume_id)
                outcome = resume_scan(
                    cfg,
                    resume_id,
                    app_factory=app_factory,
                    cancel=job.cancel,
                    on_run_created=job.set_run_id,
                    on_candidate=job.set_current_ticker,
                )
            else:
                if spec.mode == "live" and not os.environ.get("PYTEST_CURRENT_TEST"):
                    ensure_fresh_market_data(cfg, updater=update_market_data)
                if job.cancel.is_set():
                    job.mark_cancelled()
                    return
                outcome = launch_scan(
                    cfg,
                    spec,
                    app_factory=app_factory,
                    cancel=job.cancel,
                    on_run_created=job.set_run_id,
                    on_candidate=job.set_current_ticker,
                )
            job.finish(outcome)
        except Exception as exc:
            if job.cancel.is_set():
                job.mark_cancelled()
                return
            log.exception("vision scan job failed")
            job.fail(exc)

    thread = Thread(target=worker, name=f"ns-vision-scan-{job.job_id[:8]}", daemon=True)
    job.thread = thread
    thread.start()
    return job


def progress_from_job(job: ScanJob, reader: Any | None = None) -> ScanProgress:
    phase, run_id, error, outcome = job.snapshot()
    completed = pending = skipped = err_n = matches = done = total = 0
    if reader is not None and run_id:
        try:
            counts = reader.extended_counts(run_id)
            completed = int(counts.completed)
            pending = int(counts.pending)
            skipped = int(counts.skipped)
            err_n = int(counts.error)
            matches = int(counts.setup_matches)
            total = int(counts.candidates)
            done = completed + skipped + err_n
        except Exception:
            pass
    if total <= 0 and outcome is not None:
        s = outcome.summary
        completed = int(s.candidates_completed)
        pending = int(s.candidates_pending)
        skipped = int(s.candidates_skipped)
        err_n = int(s.candidates_error)
        matches = int(s.setup_matches)
        total = int(s.candidates_total)
        done = completed + skipped + err_n
    fraction = 0.0 if total <= 0 else min(1.0, done / total)
    elapsed = job.elapsed_seconds()
    active = phase in {"preparing", "running", "cancelling"}
    eta = (elapsed * (total - done) / done) if phase in {"preparing", "running"} and done > 0 and total > done else None
    ticker = job.current_ticker_name() if active else None
    if active and not ticker and reader is not None and run_id:
        ticker = _pending_ticker(reader, run_id)
    caption = format_progress_caption(
        phase=phase,
        error=error,
        ticker=ticker,
        done=done,
        total=total,
        elapsed_s=elapsed,
        eta_s=eta,
    )
    return ScanProgress(
        phase=phase,
        run_id=run_id,
        error=error,
        done=done,
        total=total,
        completed=completed,
        pending=pending,
        skipped=skipped,
        error_count=err_n,
        matches=matches,
        fraction=fraction,
        caption=caption,
        ticker=ticker,
        elapsed_seconds=elapsed,
        eta_seconds=eta,
    )


def _pending_ticker(reader: Any, run_id: str) -> str | None:
    """First unsettled candidate in scan order, used before a live ticker callback fires."""

    try:
        frozen = reader.load_frozen_inputs(run_id)
        listed = reader.store.list_candidate_results(run_id)
    except Exception:
        return None
    recorded = {r.candidate_id: r for r in listed}
    from screener_loader.vision.batching import sort_candidates

    for cand in sort_candidates(tuple(getattr(frozen, "candidates", ()) or ())):
        row = recorded.get(cand.candidate_id)
        if row is not None and getattr(row, "status", "pending") != "pending":
            continue
        ticker = str(getattr(cand, "ticker", "") or "").strip()
        if ticker:
            return ticker
    return None
