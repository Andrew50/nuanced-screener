from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
import math
import threading

import pandas as pd

from ..config import LoaderConfig
from ..duckdb_utils import connect
from .spec import ChartStyle, SetupSpec, VisionExample

_PLOT_LOCK = threading.RLock()
_OHLC_COLUMNS = ("open", "high", "low", "close")


def load_ohlcv_window(
    config: LoaderConfig,
    ticker: str,
    asof_date: date,
    lookback_bars: int,
    *,
    mask_asof_to_open_only: bool = True,
) -> pd.DataFrame:
    """Load lookback bars ending on asof_date.

    Morning-screen default: if the last row is the as-of session, keep only its
    open (premarket). High/low/close/volume of that bar are not known yet.
    Prior completed sessions stay intact.
    """
    ticker_u = str(ticker).strip().upper()
    lookback = int(lookback_bars)
    if lookback < 2:
        raise ValueError("lookback_bars must be >= 2")

    parts = config.paths.list_polygon_grouped_daily_partitions()
    con = connect(config)
    start = asof_date - timedelta(days=int(lookback) * 3 + 14)
    if parts:
        files = [p for d, p in sorted(parts.items()) if start <= d <= asof_date]
        if files:
            df = _read_files(con, files, ticker_u, asof_date)
            if not df.empty:
                return _finalize_window(df, lookback, asof_date, mask_asof_to_open_only)

    raw = config.paths.raw_ticker_parquet(ticker_u)
    if raw.exists():
        df = _read_files(con, [raw], ticker_u, asof_date)
        if not df.empty:
            return _finalize_window(df, lookback, asof_date, mask_asof_to_open_only)

    raise FileNotFoundError(
        f"No OHLCV found for {ticker_u} ending {asof_date.isoformat()}. "
        "Run `ns update` or provide per-ticker parquet."
    )


def mask_asof_bar_to_open_only(df: pd.DataFrame, asof_date: date) -> pd.DataFrame:
    """Replace the as-of session with a plottable open-only stub.

    Charts cannot render NULLs, so high/low/close equal open and volume is 0.
    No-op when the last row is an earlier completed session (weekend/holiday).
    """
    if df is None or len(df) == 0:
        return df
    work = df.copy()
    last_idx = work.index[-1]
    last_date = _as_date(work.loc[last_idx, "date"])
    if last_date != asof_date:
        return work
    open_px = pd.to_numeric(work.loc[last_idx, "open"], errors="coerce")
    if pd.isna(open_px) or not math.isfinite(float(open_px)):
        raise ValueError(f"as-of bar {asof_date.isoformat()} is missing a finite open")
    open_f = float(open_px)
    work.loc[last_idx, "high"] = open_f
    work.loc[last_idx, "low"] = open_f
    work.loc[last_idx, "close"] = open_f
    if "volume" in work.columns:
        work.loc[last_idx, "volume"] = 0.0
    return work


def _finalize_window(
    df: pd.DataFrame,
    lookback: int,
    asof_date: date,
    mask_asof_to_open_only: bool,
) -> pd.DataFrame:
    sliced = df.tail(lookback).reset_index(drop=True)
    if mask_asof_to_open_only:
        return mask_asof_bar_to_open_only(sliced, asof_date)
    return sliced


def render_chart_png(
    df: pd.DataFrame,
    *,
    ticker: str,
    asof_date: date,
    style: ChartStyle | None = None,
    title: str | None = None,
) -> bytes:
    mpl = _matplotlib()
    plt = mpl["plt"]
    fig = None
    with _PLOT_LOCK:
        try:
            fig = render_chart_figure(df, ticker=ticker, asof_date=asof_date, style=style, title=title)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
            return buf.getvalue()
        finally:
            if fig is not None:
                plt.close(fig)


def render_example_png(config: LoaderConfig, spec: SetupSpec, example: VisionExample) -> bytes:
    if example.type == "image":
        path = config.paths.setups_dir / spec.id / str(example.path)
        return Path(path).read_bytes()
    if example.date is None or not example.ticker:
        raise ValueError("market_window example is missing ticker/date")
    df = load_ohlcv_window(config, example.ticker, example.date, spec.lookback_bars)
    return render_chart_png(
        df,
        ticker=example.ticker,
        asof_date=example.date,
        style=spec.chart,
        title=f"{spec.name}  {example.ticker}  {example.date.isoformat()}",
    )


def render_chart_figure(
    df: pd.DataFrame,
    *,
    ticker: str,
    asof_date: date,
    style: ChartStyle | None = None,
    title: str | None = None,
):
    mpl = _matplotlib()
    plt = mpl["plt"]
    patches = mpl["patches"]
    style = style or ChartStyle()
    work = _prepare_ohlcv_frame(df, style=style)
    n = len(work)
    final_session = _as_date(work.loc[n - 1, "date"])
    show_vol = bool(style.volume)

    with _PLOT_LOCK:
        fig = None
        try:
            height_ratios = [3.0, 1.0] if show_vol else [1.0]
            fig, axes = plt.subplots(
                nrows=2 if show_vol else 1,
                ncols=1,
                sharex=True,
                figsize=(10.0, 6.0 if show_vol else 5.0),
                dpi=100,
                gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
                facecolor="white",
                layout="constrained",
            )
            ax = axes[0] if show_vol else axes
            vol_ax = axes[1] if show_vol else None
            if not hasattr(ax, "add_patch"):
                ax = axes

            xs = list(range(n))
            opens = work["open"].to_numpy(dtype=float)
            highs = work["high"].to_numpy(dtype=float)
            lows = work["low"].to_numpy(dtype=float)
            closes = work["close"].to_numpy(dtype=float)

            for i in xs:
                up = closes[i] >= opens[i]
                color = "#1a7f37" if up else "#c62828"
                low_i, high_i = lows[i], highs[i]
                if low_i == high_i:
                    pad = max(abs(closes[i]) * 1e-6, 1e-6)
                    low_i, high_i = low_i - pad, high_i + pad
                ax.vlines(i, low_i, high_i, color=color, linewidth=1.0, zorder=1)
                body_low = min(opens[i], closes[i])
                height = abs(closes[i] - opens[i])
                if height == 0:
                    # Doji / flat close: keep a visible wick-relative body.
                    height = max(abs(highs[i] - lows[i]) * 0.02, 1e-6)
                ax.add_patch(
                    patches.Rectangle(
                        (i - 0.3, body_low),
                        0.6,
                        height if height else 1e-6,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.6,
                        zorder=2,
                    )
                )

            for win in style.moving_averages:
                period = int(win)
                if period <= 0:
                    continue
                # Displayed bars only; min_periods=period; no pre-window warm-up.
                ma = work["close"].rolling(period, min_periods=period).mean()
                if n < period:
                    ax.plot([], [], linewidth=1.0, label=f"SMA {period} (unavailable)")
                else:
                    ax.plot(xs, ma.to_numpy(dtype=float), linewidth=1.0, label=f"SMA {period}")

            ax.set_ylabel("Price")
            ax.set_title(title or f"{ticker}  {final_session.isoformat()}")
            if style.moving_averages:
                ax.legend(loc="upper left", fontsize=8, frameon=False)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-1, n)
            _expand_flat_price_axis(ax, lows, highs)

            if vol_ax is not None:
                vols = work["volume"].to_numpy(dtype=float)
                colors = ["#1a7f37" if closes[i] >= opens[i] else "#c62828" for i in xs]
                vol_ax.bar(xs, vols, color=colors, width=0.7, align="center")
                vol_ax.set_ylabel("Vol")
                vol_ax.grid(True, alpha=0.25)
                if not vols.size or float(max(vols)) <= 0.0:
                    vol_ax.set_ylim(0.0, 1.0)

            step = max(1, n // 6)
            ticks = list(range(0, n, step))
            if ticks[-1] != n - 1:
                ticks.append(n - 1)
            labels = [work.loc[i, "date"].strftime("%Y-%m-%d") for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            return fig
        except Exception:
            if fig is not None:
                plt.close(fig)
            raise


def _prepare_ohlcv_frame(df: pd.DataFrame, *, style: ChartStyle) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("No bars to chart")
    missing = [c for c in ("date", *_OHLC_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"Chart window missing columns {missing}")
    if style.volume and "volume" not in df.columns:
        raise ValueError("Chart profile requests volume but the window has no volume column")
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    if work["date"].isna().any():
        raise ValueError("Chart window has missing dates")
    work = work.sort_values("date").reset_index(drop=True)
    for col in _OHLC_COLUMNS:
        vals = pd.to_numeric(work[col], errors="coerce")
        if vals.isna().any():
            raise ValueError(f"Chart window has non-finite {col} values")
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError(f"Chart window has non-finite {col} values")
        work[col] = vals.astype(float)
    if style.volume:
        vols = pd.to_numeric(work["volume"], errors="coerce")
        if vols.isna().any() or not all(math.isfinite(float(v)) for v in vols):
            raise ValueError("Chart window has non-finite volume values")
        work["volume"] = vols.astype(float)
    return work


def _expand_flat_price_axis(ax, lows, highs) -> None:
    lo = float(min(lows))
    hi = float(max(highs))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return
    if hi <= lo:
        pad = max(abs(lo) * 0.01, 0.01)
        ax.set_ylim(lo - pad, hi + pad)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def _read_files(con, files: list[Path], ticker: str, asof_date: date) -> pd.DataFrame:
    quoted = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return con.execute(
        f"""
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS date,
          CAST(open AS DOUBLE) AS open,
          CAST(high AS DOUBLE) AS high,
          CAST(low AS DOUBLE) AS low,
          CAST(close AS DOUBLE) AS close,
          CAST(volume AS DOUBLE) AS volume
        FROM read_parquet([{quoted}])
        WHERE UPPER(CAST(ticker AS VARCHAR)) = ?
          AND CAST(date AS DATE) <= ?
        ORDER BY date
        """,
        [ticker, asof_date],
    ).df()


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("Chart rendering requires matplotlib. Install with `pip install -e '.[ui]'`.") from e
    return {"plt": plt, "patches": patches}
