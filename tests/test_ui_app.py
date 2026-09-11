from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_shared_app_configures_pages_once() -> None:
    from screener_loader.ui import app as shared
    from screener_loader.ui import setup_builder
    from screener_loader.vision_ui import results_page

    assert "set_page_config" in inspect.getsource(shared.main)
    assert "set_page_config" not in inspect.getsource(setup_builder.render_builder_page)
    assert "set_page_config" in inspect.getsource(setup_builder.main)
    assert "set_page_config" not in inspect.getsource(results_page)
    chrome = inspect.getsource(shared)
    assert "stHeader" in chrome
    assert "stAppDeployButton" in chrome
    assert 'state_key("run_id")] = job.run_id' not in inspect.getsource(shared.main)


def test_shared_app_streamlit_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from screener_loader.config import LoaderConfig
    from screener_loader.paths import ensure_dirs
    from screener_loader.setups.service import SetupService, update_spec_fields
    from screener_loader.setups.spec import SetupFilters

    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_VISION_PAGE", "builder")
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    svc = SetupService(cfg.paths)
    spec = svc.create("Flag")
    svc.save(
        update_spec_fields(
            spec,
            filters=SetupFilters(min_dollar_vol_20d=5_000_000.0, min_adr_pct_20=0.04),
        )
    )

    app_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "ui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception
    labels = [str(getattr(r, "options", r)) for r in at.radio]
    assert any("Setups" in x and "Screener" in x for x in labels)
    assert any("Flag" in b.label for b in at.button)
    assert not any(b.label == "Create" for b in at.button)
    assert not any(getattr(t, "label", "") == "Ticker" for t in at.text_input)
    assert any(n.label == "Bars" for n in at.number_input)
    assert any(n.label == "Million USD" for n in at.number_input)
    assert any(n.label == "ADR %" for n in at.number_input)

    plus_setup = next(b for b in at.button if b.key == "sb_plus_setup")
    plus_setup.click().run()
    assert not at.exception
    assert any(b.label == "Create" for b in at.button)

    plus_ex = next(b for b in at.button if b.key == "sb_plus_example")
    plus_ex.click().run()
    assert not at.exception
    assert any(getattr(t, "label", "") == "Ticker" for t in at.text_input)



def test_filter_display_units() -> None:
    from screener_loader.ui.setup_builder import (
        adr_fraction_to_pct,
        adr_pct_to_fraction,
        format_millions_usd,
        millions_to_usd,
        usd_to_millions,
    )

    assert usd_to_millions(5_000_000) == 5.0
    assert millions_to_usd(5.0) == 5_000_000
    assert adr_fraction_to_pct(0.04) == 4.0
    assert adr_pct_to_fraction(4.0) == 0.04
    assert format_millions_usd(5_000_000.0) == "$5M"
    assert format_millions_usd(None) == "—"

