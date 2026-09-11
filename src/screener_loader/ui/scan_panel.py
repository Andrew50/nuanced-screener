"""Explicit scan launcher mounted on the shared Results page.

``render_results_page`` stays read-only: it does not import the scanner or
client, and Refresh never starts a run. This panel is the only UI path that
launches ``VisionApp``. It never auto-runs on rerender. Live mode still
requires ``NS_VISION_MODEL`` and ``OPENAI_API_KEY``; a missing key does not
fall back to demo results.

The default control is a single Run scan of the whole eligible universe.
Mode, candidate cap, and resume live under Debug.

Scans run on a background thread. The Results page polls the filesystem store
so candidate rows appear as batches commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os

from screener_loader.config import LoaderConfig
from screener_loader.vision.adapters.demo import DEMO_MODEL_ID
from screener_loader.vision.service import VisionApp, require_explicit_live_model
from screener_loader.vision.types import ScanConfig, ScanOutcome, VisionError

DRY_RUN_MODEL = "dry-run-model"
MODE_LABELS = ("Live", "Dry-run", "Demo")
MODE_BY_LABEL = {"Dry-run": "dry_run", "Demo": "demo", "Live": "live"}
DEFAULT_MAX_CANDIDATES = 0
MAX_UI_CANDIDATES = 100_000

AppFactory = Callable[..., Any]
OnRunCreated = Callable[[str], None]

_STREAMLIT = None


def _st():
    global _STREAMLIT
    if _STREAMLIT is None:
        import streamlit as st

        _STREAMLIT = st
    return _STREAMLIT


@dataclass(frozen=True)
class ScanLaunchSpec:
    mode: str
    model: str
    max_candidates: int | None


def resolve_scan_launch(
    *,
    mode: str,
    model: str,
    env_model: str = "",
    max_candidates: int,
) -> ScanLaunchSpec:
    chosen = str(mode).strip().lower()
    if chosen not in {"live", "dry_run", "demo"}:
        raise VisionError("mode must be live, dry_run, or demo")
    cap = int(max_candidates)
    if cap < 0:
        raise VisionError("max candidates cannot be negative")
    if cap > MAX_UI_CANDIDATES:
        raise VisionError(f"UI max candidates cannot exceed {MAX_UI_CANDIDATES}")
    limited = None if cap == 0 else cap
    explicit = str(model or "").strip() or str(env_model or "").strip()
    if chosen == "dry_run":
        model_id = explicit or DRY_RUN_MODEL
    elif chosen == "demo":
        model_id = explicit or DEMO_MODEL_ID
    else:
        if not explicit:
            raise VisionError("Live scans require an explicit model id (NS_VISION_MODEL)")
        model_id = require_explicit_live_model(explicit)
    return ScanLaunchSpec(mode=chosen, model=model_id, max_candidates=limited)


def format_scan_error(exc: BaseException) -> str:
    hint = getattr(exc, "rebuild_hint", None)
    text = str(exc)
    if hint:
        return f"{text} Rebuild with `{hint}`."
    return text


def _run_kwargs(*, cancel: Any | None, on_run_created: OnRunCreated | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cancel is not None:
        kwargs["cancel"] = cancel
    if on_run_created is not None:
        kwargs["on_run_created"] = on_run_created
    return kwargs


def launch_scan(
    cfg: LoaderConfig,
    spec: ScanLaunchSpec,
    *,
    app_factory: AppFactory | None = None,
    timeout_seconds: float = 60.0,
    batch_size: int = 10,
    max_output_tokens: int = 4096,
    cancel: Any | None = None,
    on_run_created: OnRunCreated | None = None,
) -> ScanOutcome:
    if spec.mode == "live" and not str(os.environ.get("OPENAI_API_KEY") or "").strip():
        raise VisionError(
            "OPENAI_API_KEY is missing; refusing to invent results. Use Dry-run or Demo for offline checks."
        )
    factory = app_factory or VisionApp
    app = factory(cfg, max_candidates=spec.max_candidates)
    config = ScanConfig(
        model=spec.model,
        mode=spec.mode,  # type: ignore[arg-type]
        timeout_seconds=float(timeout_seconds),
        batch_size=int(batch_size),
        max_output_tokens=int(max_output_tokens),
    )
    return app.run(config, **_run_kwargs(cancel=cancel, on_run_created=on_run_created))


def resume_scan(
    cfg: LoaderConfig,
    run_id: str,
    *,
    app_factory: AppFactory | None = None,
    cancel: Any | None = None,
    on_run_created: OnRunCreated | None = None,
) -> ScanOutcome:
    rid = str(run_id or "").strip()
    if not rid:
        raise VisionError("Select a saved run before resume")
    factory = app_factory or VisionApp
    return factory(cfg).resume(rid, **_run_kwargs(cancel=cancel, on_run_created=on_run_created))


def _stop_following_job() -> None:
    from screener_loader.ui.scan_jobs import FOLLOW_JOB_KEY

    _st().session_state[FOLLOW_JOB_KEY] = False


def render_scan_panel(
    cfg: LoaderConfig,
    *,
    app_factory: AppFactory | None = None,
) -> Any:
    """Start/resume a background scan. Returns the session ``ScanJob`` if one exists."""

    st = _st()
    from screener_loader.ui.scan_jobs import FOLLOW_JOB_KEY, JOB_SESSION_KEY, get_job, start_scan_job
    from screener_loader.vision.service import default_scan_root
    from screener_loader.vision.store import open_vision_scans
    from screener_loader.vision_ui.components import VIEW_ALL, state_key

    job = get_job(st.session_state.get(JOB_SESSION_KEY))
    busy = bool(job is not None and job.is_active())
    env_model = str(os.environ.get("NS_VISION_MODEL") or "")

    if "vs_mode" not in st.session_state:
        st.session_state["vs_mode"] = "Live"
    if "vs_max_candidates" not in st.session_state:
        st.session_state["vs_max_candidates"] = DEFAULT_MAX_CANDIDATES

    run_clicked = st.button("Run scan", type="primary", key="vs_run", disabled=busy, width="stretch")
    resume_clicked = False
    with st.expander("Debug"):
        st.radio("Mode", MODE_LABELS, horizontal=True, key="vs_mode")
        st.number_input(
            "Max candidates",
            min_value=0,
            max_value=MAX_UI_CANDIDATES,
            step=1,
            key="vs_max_candidates",
            help="0 = whole eligible universe. Cap after eligibility for smoke tests.",
        )
        _store, reader, _reviews = open_vision_scans(default_scan_root(cfg))
        runs = reader.list_runs()
        if runs:
            run_labels = {
                run.run_id: (
                    f"{run.created_at.strftime('%Y-%m-%d %H:%M')} · {run.status}"
                    f"{' · demo' if run.synthetic else ''} · {run.run_id[-8:]}"
                )
                for run in runs
            }
            run_ids = [r.run_id for r in runs]
            if (
                st.session_state.get(FOLLOW_JOB_KEY)
                and job is not None
                and job.run_id
            ):
                if job.run_id not in run_ids:
                    run_ids = [job.run_id, *run_ids]
                    run_labels[job.run_id] = job.run_id
                st.session_state[state_key("run_id")] = job.run_id
            elif state_key("run_id") not in st.session_state or st.session_state[state_key("run_id")] not in run_ids:
                st.session_state[state_key("run_id")] = run_ids[0]
            st.selectbox(
                "Run",
                options=run_ids,
                format_func=lambda rid, labels=run_labels: labels.get(rid, rid),
                key=state_key("run_id"),
                on_change=_stop_following_job,
            )
            if st.button("Refresh", key=state_key("refresh"), width="stretch"):
                st.rerun()
        resume_id = str(st.session_state.get(state_key("run_id")) or "").strip()
        resume_clicked = st.button(
            "Resume",
            key="vs_resume",
            disabled=busy or not resume_id,
            width="stretch",
            help=f"Resume {resume_id}" if resume_id else "Select a saved run first",
        )
    mode = MODE_BY_LABEL[str(st.session_state.get("vs_mode") or "Live")]
    max_candidates = int(st.session_state.get("vs_max_candidates") or 0)
    resume_id = str(st.session_state.get(state_key("run_id")) or "").strip()
    if run_clicked or resume_clicked:
        try:
            spec = resolve_scan_launch(
                mode=mode,
                model="",
                env_model=env_model,
                max_candidates=max_candidates,
            )
            if spec.mode == "live" and not str(os.environ.get("OPENAI_API_KEY") or "").strip():
                raise VisionError(
                    "OPENAI_API_KEY is missing; refusing to invent results. "
                    "Use Dry-run or Demo for offline checks."
                )
            started = start_scan_job(
                cfg,
                spec,
                resume_id=resume_id if resume_clicked else None,
                app_factory=app_factory,
            )
        except Exception as exc:
            st.error(format_scan_error(exc))
            return job
        st.session_state[JOB_SESSION_KEY] = started.job_id
        st.session_state[FOLLOW_JOB_KEY] = True
        st.session_state[state_key("view")] = VIEW_ALL
        st.session_state[state_key("page")] = 1
        st.session_state[state_key("candidate_id")] = None
        st.rerun()
    return get_job(st.session_state.get(JOB_SESSION_KEY))


def render_scan_status(job: Any, reader: Any | None = None) -> None:
    """Always-visible progress strip while a UI scan is in flight or just finished."""

    if job is None:
        return
    st = _st()
    from screener_loader.ui.scan_jobs import progress_from_job

    prog = progress_from_job(job, reader)
    if prog.error:
        st.error(prog.caption)
        return
    st.progress(prog.fraction if prog.total else (0.05 if job.is_active() else 1.0))
    st.caption(prog.caption)
    if job.is_active():
        if st.button("Cancel scan", key="vs_cancel"):
            job.cancel.set()
            st.rerun()
