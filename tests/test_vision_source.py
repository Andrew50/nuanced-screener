from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from screener_loader.setups.prompt import compile_prompt
from screener_loader.setups.spec import GlobalFilters, MarketCapUnavailableError, SetupFilters
from screener_loader.setups.service import update_spec_fields
from screener_loader.vision.adapters.freshness import (
    authorized_daily_session,
    expected_completed_session,
    resolve_scan_session,
)
from screener_loader.calendar_utils import calendar_local_date
from screener_loader.vision.adapters.source import SetupScanSource, capture_source_revision
from screener_loader.vision.snapshots import rule_id
from screener_loader.vision.types import (
    ChartProfileConflictError,
    InsufficientLastNError,
    ScanConfig,
    StaleSnapshotError,
    VisionError,
)
from vision_catalog import (
    ASOF,
    LOOKBACK,
    after_close_clock,
    build_catalog,
    write_last_n,
    write_raw_window,
    write_universe,
)


def _source(handles, **kwargs) -> SetupScanSource:
    return SetupScanSource(handles.cfg, clock=kwargs.pop("clock", after_close_clock), **kwargs)


def _cfg() -> ScanConfig:
    return ScanConfig(model="dry-run-model", mode="dry_run")


def test_resolve_scan_session_offline_modes_use_snapshot_max() -> None:
    expected = date(2026, 9, 9)
    snapshot = date(2026, 9, 8)
    assert resolve_scan_session(mode="dry_run", expected=expected, max_row_date=snapshot) == snapshot
    assert resolve_scan_session(mode="demo", expected=expected, max_row_date=snapshot) == snapshot
    assert resolve_scan_session(mode="live", expected=expected, max_row_date=expected) == expected
    with pytest.raises(StaleSnapshotError, match="stale"):
        resolve_scan_session(mode="live", expected=expected, max_row_date=snapshot)
    with pytest.raises(StaleSnapshotError, match="future"):
        resolve_scan_session(mode="dry_run", expected=expected, max_row_date=date(2026, 9, 10))
    assert (
        resolve_scan_session(
            mode="live",
            expected=expected,
            authorized=snapshot,
            max_row_date=snapshot,
        )
        == snapshot
    )
    with pytest.raises(StaleSnapshotError, match="stale"):
        resolve_scan_session(
            mode="live",
            expected=expected,
            authorized=snapshot,
            max_row_date=date(2026, 9, 7),
        )


def test_tickers_for_window_load_caps_before_last_n() -> None:
    from screener_loader.vision.adapters.source import tickers_for_window_load

    members = ("ZZZ", "AAA", "MMM")
    assert tickers_for_window_load(members, None) == ("AAA", "MMM", "ZZZ")
    assert tickers_for_window_load(members, 1, oversample=1) == ("AAA",)
    assert tickers_for_window_load(members, 1, oversample=4) == ("AAA", "MMM", "ZZZ")
    assert tickers_for_window_load(members, 0) == ("AAA", "MMM", "ZZZ")


def test_expected_session_uses_regular_and_early_closes() -> None:
    after = datetime(2026, 6, 18, 21, 0, tzinfo=timezone.utc)
    assert expected_completed_session(after) == date(2026, 6, 18)
    before = datetime(2026, 6, 18, 19, 0, tzinfo=timezone.utc)
    assert expected_completed_session(before) == date(2026, 6, 17)
    early_after = datetime(2025, 11, 28, 18, 30, tzinfo=timezone.utc)
    assert expected_completed_session(early_after) == date(2025, 11, 28)
    early_before = datetime(2025, 11, 28, 17, 50, tzinfo=timezone.utc)
    assert expected_completed_session(early_before) == date(2025, 11, 26)
    assert authorized_daily_session(after) == date(2026, 6, 17)
    assert authorized_daily_session(before) == date(2026, 6, 17)
    # 01:50 UTC next calendar day is still 21:50 ET; Polygon skip-today is ET, not UTC.
    after_utc_midnight = datetime(2026, 9, 10, 1, 50, tzinfo=timezone.utc)
    assert calendar_local_date(after_utc_midnight) == date(2026, 9, 9)
    assert authorized_daily_session(after_utc_midnight) == date(2026, 9, 8)
    assert expected_completed_session(after_utc_midnight) == date(2026, 9, 9)
    live_prev = resolve_scan_session(
        mode="live",
        expected=date(2026, 6, 18),
        authorized=date(2026, 6, 17),
        max_row_date=date(2026, 6, 17),
    )
    assert live_prev == date(2026, 6, 17)


def test_live_prepare_accepts_polygon_delayed_session_after_utc_midnight(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=False)
    asof = date(2026, 9, 8)
    write_last_n(
        handles.cfg,
        tickers={"AAPL": {"close": 10.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": LOOKBACK, "end": asof}},
        lookback=LOOKBACK,
        asof=asof,
    )
    write_universe(handles.cfg, ["AAPL"])
    prepared = _source(
        handles,
        clock=lambda: datetime(2026, 9, 10, 1, 50, tzinfo=timezone.utc),
    ).prepare(ScanConfig(model="gpt-4.1", mode="live"))
    assert {c.ticker for c in prepared.candidates} == {"AAPL"}
    assert {c.asof_date for c in prepared.candidates} == {asof}


def test_prepare_compatible_nondefault_profile_and_snapshots(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, lookback=LOOKBACK, moving_averages=(10, 20))
    prepared = _source(handles).prepare(_cfg())
    assert prepared.profile.lookback_bars == LOOKBACK
    assert prepared.profile.lookback_bars != 100
    assert prepared.profile.moving_averages == (10, 20)
    assert prepared.profile.volume is True
    flag = next(s for s in prepared.setups if s.setup_id == "flag")
    assert [r.rule_id for r in flag.rules] == [
        "flag:required:0",
        "flag:required:1",
        "flag:preferred:0",
        "flag:disqualifier:0",
    ]
    assert rule_id("flag", "required", 0) == "flag:required:0"
    examples = handles.service.load_examples("flag")
    spec = handles.service.get("flag")
    assert tuple(e.example_id for e in prepared.examples if e.setup_id == "flag") == compile_prompt(
        spec, examples
    ).example_ids
    upload = next(e for e in prepared.examples if e.example_id == "upload")
    assert upload.image_bytes
    assert upload.type == "image"
    assert upload.note == "uploaded bytes"
    yaml_on_disk = (tmp_path / "data" / "setups" / "flag" / "setup.yaml").read_bytes()
    assert flag.yaml_bytes == yaml_on_disk
    tickers = {c.ticker for c in prepared.candidates}
    assert "OUT" not in tickers
    assert "NA" in tickers
    assert "NVDA" in tickers
    nvda = next(c for c in prepared.candidates if c.ticker == "NVDA")
    assert nvda.eligible_setup_ids == ("ep", "flag") or set(nvda.eligible_setup_ids) == {"ep", "flag"}
    aapl = next(c for c in prepared.candidates if c.ticker == "AAPL")
    assert aapl.eligible_setup_ids == ("flag",)
    assert aapl.features.adr_pct_20 == pytest.approx(0.05)
    assert prepared.diagnostics.universe_count == 7
    assert prepared.diagnostics.not_eligible_count is None
    assert "not_eligible" in prepared.diagnostics.unavailable
    assert all(c.window.bar_count == LOOKBACK for c in prepared.candidates)
    assert all(c.window.provenance in {"unknown", "test"} for c in prepared.candidates)


def test_profile_conflicts_are_named(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, lookback=20, second_lookback=40, include_examples=False)
    with pytest.raises(ChartProfileConflictError) as exc:
        _source(handles).prepare(_cfg())
    text = str(exc.value)
    assert "lookback_bars" in text
    assert "flag=20" in text and "ep=40" in text


def test_universe_intersection_and_null_adr(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=False)
    loose = handles.service.get("flag")
    handles.service.save(update_spec_fields(loose, filters=SetupFilters()))
    prepared = _source(handles).prepare(_cfg())
    tickers = {c.ticker for c in prepared.candidates}
    assert "NULLADR" in tickers
    null = next(c for c in prepared.candidates if c.ticker == "NULLADR")
    assert null.features.adr_pct_20 is None
    assert "OUT" not in tickers
    counts = dict(prepared.diagnostics.per_setup_eligible_counts or ())
    assert counts["flag"] >= 1


def test_market_cap_fail_closed(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=False, market_cap=True)
    with pytest.raises(MarketCapUnavailableError):
        _source(handles).prepare(_cfg())


def test_globally_stale_versus_isolated_stale_and_short(tmp_path: Path) -> None:
    handles = build_catalog(
        tmp_path,
        include_examples=False,
        extra_last_n={
            "SHORT": {"close": 11.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": 5, "end": ASOF},
            "STALE": {"close": 11.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": LOOKBACK, "end": date(2026, 6, 17)},
        },
    )
    prepared = _source(handles).prepare(_cfg())
    kinds = {s.ticker: s.kind for s in prepared.skips}
    assert kinds["SHORT"] == "short_window"
    assert kinds["STALE"] == "stale"
    short = next(s for s in prepared.skips if s.ticker == "SHORT")
    aapl = next(c for c in prepared.candidates if c.ticker == "AAPL")
    if short.features is not None and short.features.close is not None:
        assert short.features.close != aapl.features.close
    assert {c.ticker for c in prepared.candidates}.isdisjoint({"SHORT", "STALE"})

    stale_all = {
        "AAPL": {"close": 10.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": LOOKBACK, "end": date(2026, 6, 16)},
    }
    write_last_n(handles.cfg, tickers=stale_all, lookback=LOOKBACK, asof=date(2026, 6, 16))
    write_universe(handles.cfg, ["AAPL"])
    dry = _source(handles).prepare(_cfg())
    assert {c.ticker for c in dry.candidates} == {"AAPL"}
    assert {c.asof_date for c in dry.candidates} == {date(2026, 6, 16)}
    with pytest.raises(StaleSnapshotError, match="stale") as live_exc:
        _source(handles).prepare(ScanConfig(model="gpt-4.1", mode="live"))
    assert "rebuild-last100" in str(getattr(live_exc.value, "rebuild_hint", ""))


def test_future_snapshot_and_insufficient_last_n(tmp_path: Path) -> None:
    handles = build_catalog(
        tmp_path,
        include_examples=False,
        extra_last_n={
            "AAPL": {"close": 10.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": LOOKBACK, "end": date(2026, 6, 19)},
        },
    )
    with pytest.raises(StaleSnapshotError, match="future"):
        _source(handles).prepare(_cfg())

    write_last_n(
        handles.cfg,
        tickers={"AAPL": {"close": 10.0, "dollar_vol": 8_000_000, "adr": 0.05, "n": 8, "end": ASOF}},
        lookback=8,
        asof=ASOF,
    )
    with pytest.raises(InsufficientLastNError) as exc:
        _source(handles).prepare(_cfg())
    assert exc.value.needed_bars == LOOKBACK
    assert "rebuild-last100" in str(exc.value)
    assert LOOKBACK == 20


def test_empty_scan_and_source_revision(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=False)
    handles.service.save_global_filters(GlobalFilters(min_price=10_000.0, min_dollar_vol_20d=1.0))
    prepared = _source(handles).prepare(_cfg())
    assert prepared.candidates == ()
    assert prepared.diagnostics.eligible_union_count == 0

    first = capture_source_revision(handles.cfg)
    yaml_path = tmp_path / "data" / "setups" / "flag" / "setup.yaml"
    yaml_path.write_bytes(yaml_path.read_bytes() + b"\n")
    second = capture_source_revision(handles.cfg)
    assert first.mismatch_message(second)


def test_example_missing_history_fails_before_scan(tmp_path: Path) -> None:
    handles = build_catalog(tmp_path, include_examples=False)
    handles.service.add_market_window_example(
        "flag",
        ticker="IBM",
        asof_date=date(2026, 5, 1),
        polarity="positive",
        quality="canonical",
    )
    write_raw_window(handles.cfg, "IBM", date(2026, 5, 1), 3)
    with pytest.raises(VisionError, match="exactly"):
        _source(handles).prepare(_cfg())
