"""Thin chart wrapper. All candlesticks/volume/SMA stay in setups.charts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import get_context
from threading import Lock
from typing import Any, Mapping
import os

from .serialization import digest_bytes
from .snapshots import moving_average_availability, profile_from_style
from .types import (
    BarWindow,
    ChartArtifact,
    ChartProfile,
    ExampleInput,
    RenderedImage,
    VisionError,
)


def png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise VisionError("Image is not a PNG")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def sniff_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def profile_to_style(profile: ChartProfile):
    from ..setups.spec import ChartStyle

    return ChartStyle(volume=bool(profile.volume), moving_averages=tuple(profile.moving_averages))


def bars_to_frame(window: BarWindow):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for bar in window.bars:
        rows.append(
            {
                "ticker": window.ticker,
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
    return pd.DataFrame(rows)


def chart_process_count() -> int:
    """Worker processes for scan rendering. ``NS_CHART_PROCESSES=1`` stays in-process."""

    raw = os.environ.get("NS_CHART_PROCESSES", "4")
    try:
        count = int(raw)
    except ValueError:
        count = 4
    return max(1, count)


def render_window_png(
    window: BarWindow,
    profile: ChartProfile,
    *,
    title: str | None = None,
) -> bytes:
    from ..setups.charts import render_chart_png

    if not window.bars:
        raise VisionError("No bars to chart")
    df = bars_to_frame(window)
    asof = window.asof_date or date.today()
    style = profile_to_style(profile)
    default_title = f"{window.ticker}  {asof.isoformat()}"
    return render_chart_png(
        df,
        ticker=window.ticker,
        asof_date=asof,
        style=style,
        title=title or default_title,
    )


def _render_window_job(window: BarWindow, profile: ChartProfile, title: str | None) -> bytes:
    """Process-pool entry point. Must stay importable from this module."""

    return render_window_png(window, profile, title=title)


class MatplotlibChartRenderer:
    """Delegates plotting to setups.charts. Does not query data or write files.

    ``open()`` starts a spawn-based process pool (default 4). Matplotlib's plot
    lock serializes threads inside one process, so scan batches submit renders
    here and the model calls stay on the scanner's thread pool. ``render()``
    without ``open()`` draws in-process.
    """

    def __init__(self, *, processes: int | None = None) -> None:
        self.processes = chart_process_count() if processes is None else max(1, int(processes))
        self._executor: ProcessPoolExecutor | None = None
        self._executor_lock = Lock()

    def open(self) -> None:
        if self.processes <= 1:
            return
        with self._executor_lock:
            if self._executor is not None:
                return
            self._executor = ProcessPoolExecutor(
                max_workers=self.processes,
                mp_context=get_context("spawn"),
            )

    def close(self) -> None:
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def render(
        self,
        window: BarWindow,
        profile: ChartProfile,
        *,
        title: str | None = None,
    ) -> ChartArtifact:
        png = self._render_png(window, profile, title=title)
        width, height = png_dimensions(png)
        asof = window.asof_date
        image = RenderedImage(
            png_bytes=png,
            width=width,
            height=height,
            sha256=digest_bytes(png),
            media_type="image/png",
        )
        return ChartArtifact(
            artifact_id=f"candidate:{window.ticker}:{asof.isoformat() if asof else 'none'}:{image.sha256[7:19]}",
            kind="candidate",
            image=image,
            title=title or f"{window.ticker}  {asof.isoformat() if asof else ''}".strip(),
            profile=profile,
            ma_availability=moving_average_availability(window.bar_count, profile.moving_averages),
            final_session_date=asof,
            ticker=window.ticker,
            source="rendered",
        )

    def _render_png(
        self,
        window: BarWindow,
        profile: ChartProfile,
        *,
        title: str | None,
    ) -> bytes:
        executor = self._executor
        if executor is None:
            return render_window_png(window, profile, title=title)
        return executor.submit(_render_window_job, window, profile, title).result()

    def wrap_upload(self, example: ExampleInput) -> ChartArtifact:
        data = example.image_bytes
        if not data:
            raise VisionError(f"upload example {example.scoped_id} has no image bytes")
        media = sniff_media_type(data)
        width, height = (0, 0)
        if media == "image/png":
            width, height = png_dimensions(data)
        image = RenderedImage(
            png_bytes=data,
            width=width,
            height=height,
            sha256=digest_bytes(data),
            media_type=media,
        )
        return ChartArtifact(
            artifact_id=f"example:{example.scoped_id}:{image.sha256[7:19]}",
            kind="example",
            image=image,
            title=example.scoped_id,
            profile=None,
            ma_availability=(),
            final_session_date=example.asof_date,
            setup_id=example.setup_id,
            example_id=example.example_id,
            ticker=example.ticker,
            source="upload",
        )


def chart_style_profile(style, *, timeframe: str = "1d", lookback_bars: int) -> ChartProfile:
    return profile_from_style(style, timeframe=timeframe, lookback_bars=lookback_bars)


def frame_from_mappings(rows: list[Mapping[str, Any]]):
    import pandas as pd

    return pd.DataFrame(list(rows))
