from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from screener_loader.config import LoaderConfig
from screener_loader.ui.scan_panel import (
    DEMO_MODEL_ID,
    DRY_RUN_MODEL,
    format_scan_error,
    launch_scan,
    resolve_scan_launch,
)
from screener_loader.vision.types import (
    ScanDiagnostics,
    ScanOutcome,
    RunSummary,
    TokenUsage,
    VisionError,
)


def test_resolve_scan_launch_fills_offline_placeholders() -> None:
    dry = resolve_scan_launch(mode="dry_run", model="", max_candidates=5)
    assert dry.mode == "dry_run"
    assert dry.model == DRY_RUN_MODEL
    demo = resolve_scan_launch(mode="demo", model="", max_candidates=3)
    assert demo.mode == "demo"
    assert demo.model == DEMO_MODEL_ID
    assert demo.max_candidates == 3


def test_resolve_scan_launch_live_requires_real_model() -> None:
    with pytest.raises(VisionError, match="explicit model"):
        resolve_scan_launch(mode="live", model="", env_model="", max_candidates=5)
    with pytest.raises(VisionError, match="demo/dry-run"):
        resolve_scan_launch(mode="live", model=DEMO_MODEL_ID, max_candidates=5)
    live = resolve_scan_launch(mode="live", model="", env_model="gpt-4.1", max_candidates=8)
    assert live.model == "gpt-4.1"
    assert live.max_candidates == 8


def test_resolve_scan_launch_zero_is_uncapped() -> None:
    spec = resolve_scan_launch(mode="demo", model="", max_candidates=0)
    assert spec.max_candidates is None
    with pytest.raises(VisionError, match="negative"):
        resolve_scan_launch(mode="demo", model="", max_candidates=-1)


def test_format_scan_error_appends_rebuild_hint() -> None:
    class _E(Exception):
        rebuild_hint = "ns rebuild-last100 --window-size 120"

    assert "Rebuild with" in format_scan_error(_E("short"))


def test_launch_scan_live_refuses_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spec = resolve_scan_launch(mode="live", model="gpt-4.1", max_candidates=2)

    def _factory(*_a, **_k):
        raise AssertionError("VisionApp must not start when the key is missing")

    with pytest.raises(VisionError, match="OPENAI_API_KEY"):
        launch_scan(LoaderConfig(repo_root=tmp_path), spec, app_factory=_factory)


def test_launch_scan_uses_factory_and_cap(tmp_path: Path) -> None:
    spec = resolve_scan_launch(mode="demo", model="", max_candidates=4)
    seen: dict[str, object] = {}

    @dataclass
    class _FakeApp:
        def __init__(self, _cfg, *, max_candidates=None):
            seen["max_candidates"] = max_candidates

        def run(self, config, **_kwargs):
            seen["mode"] = config.mode
            seen["model"] = config.model
            return ScanOutcome(
                run_id="run-ui-test",
                status="completed",
                summary=RunSummary(
                    candidates_total=1,
                    candidates_completed=1,
                    candidates_error=0,
                    candidates_skipped=0,
                    candidates_pending=0,
                    setup_matches=1,
                    attempts=1,
                    usage=TokenUsage(1, 1, 2),
                    status="completed",
                ),
                diagnostics=ScanDiagnostics(),
                semantic_digest="s",
                raw_digest="r",
            )

    out = launch_scan(LoaderConfig(repo_root=tmp_path), spec, app_factory=_FakeApp)
    assert out.run_id == "run-ui-test"
    assert seen["max_candidates"] == 4
    assert seen["mode"] == "demo"
    assert seen["model"] == DEMO_MODEL_ID


def test_results_page_still_does_not_import_scan() -> None:
    import screener_loader.vision_ui.results_page as rp

    assert "screener_loader.vision.client" not in rp.__dict__
    assert "screener_loader.vision.scan" not in rp.__dict__
    assert "render_scan_panel" not in rp.__dict__


def test_scan_panel_button_calls_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))

    class _FakeApp:
        def __init__(self, _cfg, *, max_candidates=None):
            self.max_candidates = max_candidates

        def run(self, config, **kwargs):
            assert config.mode in {"demo", "dry_run", "live"}
            cb = kwargs.get("on_run_created")
            if cb:
                cb("run-from-ui")
            return ScanOutcome(
                run_id="run-from-ui",
                status="completed",
                summary=RunSummary(
                    candidates_total=2,
                    candidates_completed=2,
                    candidates_error=0,
                    candidates_skipped=0,
                    candidates_pending=0,
                    setup_matches=1,
                    attempts=1,
                    usage=None,
                    status="completed",
                ),
                diagnostics=ScanDiagnostics(),
                semantic_digest="s",
                raw_digest="r",
            )

        def resume(self, run_id):
            raise AssertionError(f"resume should not run in this test: {run_id}")

    monkeypatch.setattr("screener_loader.ui.scan_panel.VisionApp", _FakeApp)

    def _page():
        import os
        from pathlib import Path

        import streamlit as st

        from screener_loader.config import LoaderConfig as LC
        from screener_loader.ui.scan_panel import render_scan_panel as panel

        rid = panel(LC(repo_root=Path(os.environ["NS_REPO_ROOT"])))
        if rid:
            st.write(f"launched:{rid}")

    at = AppTest.from_function(_page, default_timeout=30)
    at.run()
    assert not at.exception
    radios = [r for r in at.radio if "Dry-run" in str(getattr(r, "options", ()))]
    assert radios
    radios[0].set_value("Demo").run()
    assert not at.exception
    assert not any(getattr(t, "label", "") == "Model" for t in at.text_input)
    run_btn = next(b for b in at.button if b.label == "Run scan")
    run_btn.click().run()
    assert not at.exception
    from screener_loader.ui.scan_jobs import _JOBS, _JOBS_LOCK

    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    assert jobs, "scan job was not started"
    job = jobs[-1]
    if job.thread is not None:
        job.thread.join(timeout=5)
    assert job.run_id == "run-from-ui"
    assert job.snapshot()[0] == "done"


def test_shared_app_screener_lists_results_in_sidebar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from screener_loader.config import LoaderConfig
    from screener_loader.paths import ensure_dirs
    from screener_loader.vision.adapters.source import vision_scan_root
    from screener_loader.vision_ui.synthetic import seed_synthetic_runs

    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_VISION_PAGE", "results")
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    seed_synthetic_runs(vision_scan_root(cfg))

    app_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "ui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception
    labels = [str(b.label) for b in at.button]
    assert "Run scan" in labels
    assert "Prev page" not in labels
    assert "Next page" not in labels
    assert any(label == "Previous" for label in labels)
    assert any(label == "Next" for label in labels)
    assert any(label in {"AAPL", "AAA", "MSFT", "NVDA"} for label in labels)
    assert not any(label.startswith("▸ ") for label in labels)
    assert not any(getattr(e, "label", None) == "Export" for e in at.expander)
    assert any(getattr(e, "label", None) == "Debug" for e in at.expander)
    assert any(getattr(e, "label", None) == "Filters" for e in at.expander)


def test_shared_app_results_shows_scan_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_VISION_PAGE", "results")
    app_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "ui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception
    labels = [str(b.label) for b in at.button]
    assert "Run scan" in labels
    assert any("Resume" in label for label in labels)
    assert not any(getattr(t, "label", "") == "Model" for t in at.text_input)
    assert not any(getattr(e, "label", None) == "Run a scan" for e in at.expander)
    assert any(getattr(e, "label", None) == "Debug" for e in at.expander)
    markdown = " ".join(str(getattr(m, "value", m)) for m in at.markdown)
    assert "Nuanced" not in markdown


def _outcome(run_id: str = "run-job") -> ScanOutcome:
    return ScanOutcome(
        run_id=run_id,
        status="completed",
        summary=RunSummary(
            candidates_total=4,
            candidates_completed=2,
            candidates_error=0,
            candidates_skipped=1,
            candidates_pending=1,
            setup_matches=3,
            attempts=1,
            usage=None,
            status="completed",
        ),
        diagnostics=ScanDiagnostics(),
        semantic_digest="s",
        raw_digest="r",
    )


def test_background_job_records_run_id_live(tmp_path: Path) -> None:
    from screener_loader.ui.scan_jobs import progress_from_job, start_scan_job

    created: list[str] = []

    class _FakeApp:
        def __init__(self, _cfg, *, max_candidates=None):
            self.max_candidates = max_candidates

        def run(self, config, **kwargs):
            cb = kwargs.get("on_run_created")
            if cb:
                cb("run-live-1")
                created.append("run-live-1")
            return _outcome("run-live-1")

    spec = resolve_scan_launch(mode="demo", model="", max_candidates=4)
    job = start_scan_job(LoaderConfig(repo_root=tmp_path), spec, app_factory=_FakeApp)
    assert job.thread is not None
    job.thread.join(timeout=5)
    assert job.snapshot()[0] == "done"
    assert job.run_id == "run-live-1"
    assert created == ["run-live-1"]
    prog = progress_from_job(job)
    assert prog.completed == 2
    assert prog.matches == 3
    assert 0 < prog.fraction <= 1


def test_progress_preparing_has_caption() -> None:
    from screener_loader.ui.scan_jobs import ScanJob, progress_from_job

    job = ScanJob("abc")
    prog = progress_from_job(job)
    assert prog.phase == "preparing"
    assert "Preparing" in prog.caption


def test_bind_run_created_invokes_callback() -> None:
    from screener_loader.vision.service import VisionApp

    seen: list[str] = []

    class _Stored:
        run_id = "run-bound"

    class _Store:
        def create_run(self, _prepared):
            return _Stored()

    class _Scanner:
        def __init__(self):
            self.store = _Store()

    scanner = _Scanner()
    VisionApp._bind_run_created(scanner, seen.append)
    scanner.store.create_run(None)
    assert seen == ["run-bound"]

