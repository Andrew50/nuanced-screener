from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.prompt import compile_prompt
from screener_loader.setups.service import SetupService, update_spec_fields
from screener_loader.setups.spec import (
    MarketCapUnavailableError,
    SetupCriteria,
    SetupFilters,
    SetupValidationError,
    slugify_setup_id,
)


def _svc(tmp_path: Path) -> SetupService:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    return SetupService(cfg.paths)


def test_create_list_disable_roundtrip(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    spec = svc.create("Flag", description="Impulse then consolidation.")
    spec = svc.save(
        update_spec_fields(
            spec,
            criteria=SetupCriteria(required=("tight flag",), preferred=("volume dry-up",), disqualifiers=("extended",)),
            filters=SetupFilters(min_adr_pct_20=0.04),
        )
    )
    assert spec.id == "flag"
    assert svc.list_enabled()[0].id == "flag"
    svc.set_enabled("flag", False)
    assert svc.list_enabled() == []
    assert svc.get("flag").enabled is False
    assert (tmp_path / "data" / "setups" / "flag" / "setup.yaml").exists()


def test_examples_not_written_to_labels_csv(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    svc.create("Episodic Pivot")
    svc.add_market_window_example(
        "episodic_pivot",
        ticker="NVDA",
        asof_date=date(2026, 6, 18),
        polarity="positive",
        quality="canonical",
        note="Best example",
    )
    labels = tmp_path / "labels.csv"
    assert not labels.exists()
    examples = svc.load_examples("episodic_pivot")
    assert examples[0].ticker == "NVDA"
    assert examples[0].polarity == "positive"


def test_positive_and_negative_examples(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    svc.create("Flag")
    svc.add_market_window_example(
        "flag", ticker="NVDA", asof_date=date(2026, 6, 12), polarity="positive", quality="canonical"
    )
    svc.add_market_window_example(
        "flag",
        ticker="XYZ",
        asof_date=date(2026, 4, 1),
        polarity="negative",
        quality="near_miss",
        note="too extended",
    )
    compiled = svc.compile_prompt("flag")
    assert "NVDA 2026-06-12" in compiled.text
    assert "too extended" in compiled.text
    assert compiled.example_ids[0].startswith("nvda_")
    assert "xyz_" in compiled.example_ids[1]


def test_image_example_copied(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    svc.create("Flag")
    src = tmp_path / "shot.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
    ex = svc.add_image_example("flag", src, polarity="negative", quality="near_miss")
    dest = tmp_path / "data" / "setups" / "flag" / "examples" / f"{ex.id}.png"
    assert dest.exists()
    assert dest.read_bytes() == src.read_bytes()


def test_market_cap_fail_closed(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    spec = svc.create("Flag")
    with pytest.raises(MarketCapUnavailableError):
        svc.save(update_spec_fields(spec, filters=SetupFilters(min_market_cap=100_000_000)))


def test_setup_cannot_loosen_global(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    spec = svc.create("Flag")
    with pytest.raises(SetupValidationError, match="loosens"):
        svc.save(update_spec_fields(spec, filters=SetupFilters(min_price=0.5)))


def test_reject_intraday_timeframe(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    with pytest.raises(SetupValidationError, match="1d"):
        svc.create("Flag", timeframe="5m")


def test_compile_prompt_structure(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    spec = svc.create("Flag", description="Controlled consolidation.")
    svc.save(
        update_spec_fields(
            spec,
            criteria=SetupCriteria(required=("breakout",), preferred=("gap",), disqualifiers=("choppy",)),
        )
    )
    text = compile_prompt(svc.get("flag"), []).text
    assert "Required:" in text
    assert "- breakout" in text
    assert "- choppy" in text


def test_create_slugs_name_and_avoids_collision(tmp_path: Path) -> None:
    assert slugify_setup_id("Episodic Pivot") == "episodic_pivot"
    assert slugify_setup_id("Flag") == "flag"
    svc = _svc(tmp_path)
    a = svc.create("Episodic Pivot")
    b = svc.create("Episodic Pivot")
    assert a.id == "episodic_pivot"
    assert a.name == "Episodic Pivot"
    assert b.id == "episodic_pivot_2"
    assert b.name == "Episodic Pivot"


def test_flag_catalog_examples_are_screenshot_charts() -> None:
    from screener_loader.paths import DataPaths
    from screener_loader.setups.store import load_examples, load_setup

    paths = DataPaths(Path(__file__).resolve().parents[1])
    spec = load_setup(paths, "flag")
    examples = load_examples(paths, "flag")
    assert spec.name == "Bull Flag"
    by_id = {e.id: e for e in examples}
    assert by_id["gotu_chart"].quality == "canonical"
    assert by_id["hlx_chart"].quality == "canonical"
    assert by_id["meli_chart"].quality == "canonical"
    assert by_id["mara_chart"].quality == "edge_case"
    assert all(e.polarity == "positive" and e.type == "image" for e in examples)
    for ex in examples:
        dest = paths.setups_dir / "flag" / str(ex.path)
        assert dest.is_file()
        assert dest.stat().st_size > 1000

