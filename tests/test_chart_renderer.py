from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.charts import load_ohlcv_window, mask_asof_bar_to_open_only, render_chart_png
from screener_loader.setups.spec import ChartStyle


matplotlib = pytest.importorskip("matplotlib")


def test_chart_png_is_deterministic(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    start = date(2025, 1, 2)
    rows = []
    for i in range(30):
        d = start + timedelta(days=i)
        px = 100.0 + i
        rows.append(
            {
                "ticker": "NVDA",
                "date": d,
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.2,
                "volume": 1_000_000 + i,
                "adj_close": px,
            }
        )
    pd.DataFrame(rows).to_parquet(cfg.paths.raw_ticker_parquet("NVDA"), index=False)
    asof = start + timedelta(days=29)
    df = load_ohlcv_window(cfg, "NVDA", asof, 20)
    assert len(df) == 20
    a = render_chart_png(df, ticker="NVDA", asof_date=asof, style=ChartStyle())
    b = render_chart_png(df, ticker="NVDA", asof_date=asof, style=ChartStyle())
    assert a == b
    assert a[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_ohlcv_window_masks_asof_bar_to_open_only(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    start = date(2025, 1, 2)
    rows = []
    for i in range(10):
        d = start + timedelta(days=i)
        px = 100.0 + i
        rows.append(
            {
                "ticker": "NVDA",
                "date": d,
                "open": px,
                "high": px + 2.0,
                "low": px - 1.5,
                "close": px + 0.4,
                "volume": 1_000_000 + i,
                "adj_close": px,
            }
        )
    pd.DataFrame(rows).to_parquet(cfg.paths.raw_ticker_parquet("NVDA"), index=False)
    asof = start + timedelta(days=9)
    masked = load_ohlcv_window(cfg, "NVDA", asof, 8)
    last = masked.iloc[-1]
    prev = masked.iloc[-2]
    assert last["date"] == asof or pd.Timestamp(last["date"]).date() == asof
    assert float(last["high"]) == float(last["open"])
    assert float(last["low"]) == float(last["open"])
    assert float(last["close"]) == float(last["open"])
    assert float(last["volume"]) == 0.0
    assert float(prev["high"]) == float(prev["open"]) + 2.0
    assert float(prev["volume"]) == 1_000_000 + 8

    full = load_ohlcv_window(cfg, "NVDA", asof, 8, mask_asof_to_open_only=False)
    assert float(full.iloc[-1]["high"]) == float(full.iloc[-1]["open"]) + 2.0
    assert float(full.iloc[-1]["volume"]) == 1_000_000 + 9


def test_mask_asof_noop_when_last_bar_is_prior_session() -> None:
    prior = date(2023, 12, 21)
    asof = date(2023, 12, 22)
    df = pd.DataFrame(
        [
            {
                "date": prior,
                "open": 25.0,
                "high": 27.0,
                "low": 24.0,
                "close": 26.0,
                "volume": 5_000_000,
            }
        ]
    )
    out = mask_asof_bar_to_open_only(df, asof)
    assert float(out.iloc[-1]["high"]) == 27.0
    assert float(out.iloc[-1]["volume"]) == 5_000_000
