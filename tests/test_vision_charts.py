from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.charts import load_ohlcv_window, render_chart_figure, render_chart_png, render_example_png
from screener_loader.setups.spec import ChartStyle, VisionExample
from screener_loader.vision.charts import MatplotlibChartRenderer, png_dimensions, profile_to_style
from screener_loader.vision.snapshots import make_bar_window, bars_from_rows
from vision_support import SYNTHETIC_PNG, default_profile, make_example, make_spec, snapshot_pair

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ohlcv_df(n: int = 30, *, asof: date = date(2026, 6, 18), flat: bool = False, doji: bool = False, zero_vol: bool = False) -> pd.DataFrame:
    start = asof - timedelta(days=n - 1)
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        px = 50.0 if flat else 100.0 + i * 0.5
        o = px if doji else px - 0.2
        c = px if doji else px + 0.15
        rows.append(
            {
                "ticker": "NVDA",
                "date": d,
                "open": o,
                "high": px if flat else px + 0.8,
                "low": px if flat else px - 0.7,
                "close": c,
                "volume": 0.0 if zero_vol else 1_000_000 + i,
            }
        )
    return pd.DataFrame(rows)


def test_vision_builder_and_wrapper_use_same_plotter_and_bytes() -> None:
    df = _ohlcv_df(24)
    asof = date(2026, 6, 18)
    style = ChartStyle(volume=True, moving_averages=(10, 20, 50))
    title = "NVDA  2026-06-18"
    direct = render_chart_png(df, ticker="NVDA", asof_date=asof, style=style, title=title)
    window = make_bar_window("NVDA", bars_from_rows(df.to_dict("records")))
    profile = default_profile(lookback_bars=24, moving_averages=(10, 20, 50))
    wrapped = MatplotlibChartRenderer().render(window, profile, title=title)
    assert direct == wrapped.image.png_bytes
    assert wrapped.image.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = png_dimensions(wrapped.image.png_bytes)
    assert wrapped.image.width == w > 10
    assert wrapped.image.height == h > 10
    assert wrapped.final_session_date == asof
    assert profile_to_style(profile).moving_averages == style.moving_averages
    assert render_chart_png.__module__.endswith("setups.charts")
    import inspect

    from screener_loader.vision.charts import render_window_png

    assert "from ..setups.charts import render_chart_png" in inspect.getsource(render_window_png)


def test_vision_public_chart_signatures_still_load_and_render(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    df = _ohlcv_df(20)
    df.to_parquet(cfg.paths.raw_ticker_parquet("NVDA"), index=False)
    asof = date(2026, 6, 18)
    loaded = load_ohlcv_window(cfg, "NVDA", asof, 12)
    assert len(loaded) == 12
    png = render_chart_png(loaded, ticker="NVDA", asof_date=asof, style=ChartStyle())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    spec = make_spec("flag")
    example = VisionExample(
        id="ex1",
        polarity="positive",
        type="market_window",
        ticker="NVDA",
        date=asof,
        quality="canonical",
    )
    example_png = render_example_png(cfg, spec, example)
    assert example_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_vision_malformed_windows_fail_cleanly() -> None:
    asof = date(2026, 6, 18)
    with pytest.raises(ValueError, match="No bars"):
        render_chart_png(pd.DataFrame(), ticker="X", asof_date=asof)
    with pytest.raises(ValueError, match="missing columns"):
        render_chart_png(pd.DataFrame({"date": [asof], "close": [1.0]}), ticker="X", asof_date=asof)
    bad = _ohlcv_df(5)
    bad.loc[0, "high"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        render_chart_png(bad, ticker="X", asof_date=asof)
    before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        render_chart_png(pd.DataFrame(), ticker="X", asof_date=asof)
    assert set(plt.get_fignums()) == before


def test_vision_sma_display_only_and_unavailable_long_period() -> None:
    df = _ohlcv_df(20)
    asof = date(2026, 6, 18)
    style = ChartStyle(moving_averages=(10, 20, 50))
    before = set(plt.get_fignums())
    fig = render_chart_figure(df, ticker="NVDA", asof_date=asof, style=style)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "SMA 10" in labels
        assert "SMA 20" in labels
        assert "SMA 50 (unavailable)" in labels
        line = next(l for l in fig.axes[0].lines if l.get_label() == "SMA 10")
        ys = line.get_ydata()
        assert any(pd.isna(ys[:9])) or len(ys) == 20
        # min_periods=period: first 9 of SMA 10 are unavailable on displayed bars.
        import numpy as np

        assert np.isnan(ys[0])
        assert np.isfinite(ys[9])
    finally:
        plt.close(fig)
    assert set(plt.get_fignums()) == before


def test_vision_nondefault_profile_volume_off_and_cleanup() -> None:
    df = _ohlcv_df(40)
    asof = date(2026, 6, 18)
    style = ChartStyle(volume=False, moving_averages=(10,))
    before = set(plt.get_fignums())
    png = render_chart_png(df, ticker="NVDA", asof_date=asof, style=style, title="custom")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert set(plt.get_fignums()) == before
    window = make_bar_window("NVDA", bars_from_rows(df.to_dict("records")))
    profile = default_profile(lookback_bars=40, volume=False, moving_averages=(10,))
    art = MatplotlibChartRenderer().render(window, profile, title="custom")
    assert art.image.png_bytes == png
    assert art.ma_availability[0].available is True
    assert len(art.ma_availability) == 1


def test_vision_flat_doji_zero_volume_and_final_session_title() -> None:
    asof = date(2026, 6, 18)
    before = set(plt.get_fignums())
    flat = render_chart_png(_ohlcv_df(8, flat=True), ticker="FLAT", asof_date=asof)
    doji = render_chart_png(_ohlcv_df(8, doji=True), ticker="DOJI", asof_date=asof)
    zvol = render_chart_png(_ohlcv_df(8, zero_vol=True), ticker="ZVOL", asof_date=asof)
    assert flat[:8] == doji[:8] == zvol[:8] == b"\x89PNG\r\n\x1a\n"
    fig = render_chart_figure(_ohlcv_df(8, flat=True), ticker="FLAT", asof_date=date(1999, 1, 1))
    try:
        assert "2026-06-18" in fig.axes[0].get_title()
    finally:
        plt.close(fig)
    assert set(plt.get_fignums()) == before


def test_process_pool_png_matches_in_process_chart() -> None:
    from concurrent.futures import ThreadPoolExecutor

    df = _ohlcv_df(40)
    asof = date(2026, 6, 18)
    window = make_bar_window("NVDA", bars_from_rows(df.to_dict("records")))
    profile = default_profile(lookback_bars=40, moving_averages=(10, 20, 50))
    title = "NVDA  2026-06-18"
    inline = MatplotlibChartRenderer(processes=1).render(window, profile, title=title)
    pool = MatplotlibChartRenderer(processes=4)
    pool.open()
    try:
        assert pool._executor is not None
        assert pool._executor._max_workers == 4
        with ThreadPoolExecutor(max_workers=8) as threads:
            arts = list(threads.map(lambda _i: pool.render(window, profile, title=title), range(8)))
    finally:
        pool.close()
    assert pool._executor is None
    for art in arts:
        assert art.image.png_bytes == inline.image.png_bytes
        assert art.image.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_vision_upload_example_keeps_original_bytes() -> None:
    spec = make_spec("flag")
    example = make_example("shot", kind="image", polarity="negative")
    _, inputs = snapshot_pair(spec, [example], image_bytes=SYNTHETIC_PNG)
    art = MatplotlibChartRenderer().wrap_upload(inputs[0])
    assert art.image.png_bytes == SYNTHETIC_PNG
    assert art.source == "upload"
    assert art.kind == "example"
