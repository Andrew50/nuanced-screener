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

    def is_active(self) -> bool:
        with self._lock:
            return self.phase in {"preparing", "running"}

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

    def finish(self, outcome: ScanOutcome) -> None:
        with self._lock:
            self.outcome = outcome
            self.run_id = outcome.run_id
            self.phase = "cancelled" if self.cancel.is_set() else "done"
            self.error = None

    def fail(self, exc: BaseException) -> None:
        from screener_loader.ui.scan_panel import format_scan_error

        with self._lock:
            self.phase = "error"
            self.error = format_scan_error(exc)


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
            if resume_id:
                job.set_run_id(resume_id)
                outcome = resume_scan(
                    cfg,
                    resume_id,
                    app_factory=app_factory,
                    cancel=job.cancel,
                    on_run_created=job.set_run_id,
                )
            else:
                if spec.mode == "live" and not os.environ.get("PYTEST_CURRENT_TEST"):
                    ensure_fresh_market_data(cfg, updater=update_market_data)
                outcome = launch_scan(
                    cfg,
                    spec,
                    app_factory=app_factory,
                    cancel=job.cancel,
                    on_run_created=job.set_run_id,
                )
            job.finish(outcome)
        except Exception as exc:
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
    if error:
        caption = error
    elif phase == "preparing":
        caption = "Preparing eligibility, last-N windows, and example charts…"
    elif phase == "running":
        caption = (
            f"Scanning {run_id or 'new run'} · {done}/{total or '?'} settled · "
            f"{completed} completed · {pending} pending · {matches} setup-matches · "
            f"{skipped} skipped · {err_n} error"
        )
    elif phase == "cancelled":
        caption = f"Cancelled {run_id or ''} · {done}/{total or 0} settled"
    elif phase == "done":
        caption = (
            f"Finished {run_id or ''} · {completed} completed · {matches} setup-matches · "
            f"{skipped} skipped · {err_n} error"
        )
    else:
        caption = error or phase
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
    )
