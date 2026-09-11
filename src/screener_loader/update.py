from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import LoaderConfig
from .calendar_utils import TradingCalendar, calendar_local_date, latest_authorized_daily_session, subtract_years
from .derived import ensure_last_n_bars
from .http_client import configure_host_rate_limit
from .paths import ensure_dirs
from .raw_update import TickerResult, merge_write_ticker_parquet, plan_task_for_ticker
from .universe import build_universe, load_universe
from .update_state import write_update_state
from .vendors.registry import get_ohlcv_vendor
from .vendors.polygon_grouped import PolygonGroupedDailyVendor, require_polygon_api_key


def _chunked(items: list, n: int) -> Iterable[list]:
    if n <= 0:
        yield items
        return
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _retry_sleep(attempt: int) -> float:
    # Exponential backoff with a small cap; jitter added via time.time fractional part.
    base = min(10.0, 0.5 * (2**attempt))
    jitter = (time.time() % 1.0) * 0.2
    return base + jitter


def _download_then_merge_one(
    config: LoaderConfig,
    ticker: str,
    start,
    end,
    full_refresh: bool,
) -> TickerResult:
    vendor = get_ohlcv_vendor(config)
    vendor_symbol = vendor.normalize_symbol(ticker)

    last_err: str | None = None
    for attempt in range(0, max(1, config.max_retries) + 1):
        try:
            df = vendor.fetch_daily_ohlcv(
                vendor_symbol=vendor_symbol,
                start=start,
                end=end,
                config=config,
            )
            if df is None or df.empty:
                return merge_write_ticker_parquet(config, ticker=ticker, new_bars=df, full_refresh=full_refresh)
            df = df.copy()
            df["ticker"] = ticker
            return merge_write_ticker_parquet(config, ticker=ticker, new_bars=df, full_refresh=full_refresh)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "Invalid Crumb" in msg or "Unauthorized" in msg:
                msg = f"{msg} (Yahoo blocked/crumb issue; try --ohlcv-vendor stooq)"
            if "Connection refused" in msg or "Failed to establish a new connection" in msg:
                msg = f"{msg} (remote refused connections; try lowering --processes, e.g. 4-8)"
            last_err = f"{type(e).__name__}: {msg}"
            if attempt >= config.max_retries:
                break
            time.sleep(_retry_sleep(attempt))

    return TickerResult(ticker=ticker, status="failed", error=last_err)


def _write_ticker_state(config: LoaderConfig, results: list[TickerResult]) -> Path:
    path = config.paths.ticker_state_parquet
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        rows.append(
            {
                "ticker": r.ticker,
                "status": r.status,
                "rows_written": r.rows_written,
                "last_date": r.last_date,
                "error": r.error,
                "run_ts": now,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _write_run_log(config: LoaderConfig, results: list[TickerResult]) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = config.paths.logs_dir / f"update_{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")
    return out


def _update_market_data_per_ticker(config: LoaderConfig) -> None:
    """
    Incrementally update per-ticker Parquet history and then rebuild derived last-N dataset.

    - Tolerates partial failures (keeps going unless fail_fast=True)
    - Avoids rewriting large partitions by writing only per-ticker files that changed
    """
    ensure_dirs(config.paths)

    # Universe is required; build it automatically if missing.
    if not config.paths.tickers_csv.exists():
        print("[yellow]tickers.csv not found; building universe first[/yellow]")
        build_universe(config)

    tickers_df = load_universe(config)
    tickers = tickers_df["ticker"].astype(str).str.upper().tolist()

    tasks = []
    for t in tickers:
        task = plan_task_for_ticker(config, t)
        if task is None:
            continue
        tasks.append(task)

    if not tasks:
        print("[green]No tickers need updating[/green]")
        ensure_last_n_bars(config)
        write_update_state(config.paths, vendor=str(config.ohlcv_vendor), started_at=datetime.now(timezone.utc))
        return

    print(f"[cyan]Planning[/cyan] {len(tasks)} ticker updates (batch_size={config.batch_size}, {config.executor}={config.processes})")

    results: list[TickerResult] = []
    Executor = ThreadPoolExecutor if config.executor == "threads" else ProcessPoolExecutor

    completed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=False,
        refresh_per_second=10,
    ) as progress:
        task_id = progress.add_task("Updating tickers", total=len(tasks))

        for batch in _chunked(tasks, config.batch_size):
            if config.pause_seconds > 0 and results:
                time.sleep(config.pause_seconds)

            with Executor(max_workers=int(config.processes)) as ex:
                futs = []
                for task in batch:
                    futs.append(
                        ex.submit(
                            _download_then_merge_one,
                            config,
                            task.ticker,
                            task.start,
                            task.end,
                            task.full_refresh,
                        )
                    )

                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    completed += 1
                    progress.update(task_id, completed=completed)

                    # Keep per-ticker output (useful for diagnosing partial failures).
                    if r.status == "failed":
                        print(f"[red]FAILED[/red] {r.ticker} - {r.error}")
                        if config.fail_fast:
                            break
                    elif r.status == "updated":
                        print(f"[green]UPDATED[/green] {r.ticker} (rows={r.rows_written}, last={r.last_date})")
                    else:
                        print(f"[dim]{r.status.upper()}[/dim] {r.ticker}")

            if config.fail_fast and any(r.status == "failed" for r in results):
                break

    _write_ticker_state(config, results)
    _write_run_log(config, results)

    print("[cyan]Refreshing derived last-N cache[/cyan]")
    ensure_last_n_bars(config)
    write_update_state(
        config.paths,
        vendor=str(config.ohlcv_vendor),
        dates_updated=[r.ticker for r in results if r.status == "updated"],
        dates_failed=[r.ticker for r in results if r.status == "failed"],
        ok=not any(r.status == "failed" for r in results) or not config.fail_fast,
    )


def _plan_polygon_dates(
    config: LoaderConfig,
    cal: TradingCalendar,
    today: date,
    existing_partitions: set[date],
    now_utc: datetime | None = None,
) -> list[date]:
    end = latest_authorized_daily_session(today, cal, skip_same_calendar_day=now_utc is not None)

    start = subtract_years(end, config.lookback_years)
    trading_days = cal.valid_trading_days(start, end)
    if not trading_days:
        return []

    trading_set = set(trading_days)
    existing_in_window = existing_partitions & trading_set
    missing_in_window = trading_set - existing_partitions

    # Required ordering:
    # 1) Fetch the most recent trading day first, always (overwrite if it exists).
    # 2) Then fetch days that have never been loaded (missing), newest -> oldest.
    # 3) Then re-fetch already-loaded days, newest -> oldest.
    #    Daily runs only refresh `refresh_tail_days` (default 3) so cron stays cheap.
    #    `--full-refresh` re-pulls every existing partition in the lookback window.
    #
    # Notes:
    # - We never delete partitions. Anything outside the lookback window is left intact
    #   (and cannot be re-fetched here anyway).
    ordered: list[date] = []

    # Phase 1: latest day always.
    ordered.append(end)

    # Phase 2: missing days in window, newest -> oldest.
    for d in sorted(missing_in_window, reverse=True):
        ordered.append(d)

    # Phase 3: already-loaded days in window, newest -> oldest.
    existing_sorted = sorted(existing_in_window, reverse=True)
    if config.full_refresh:
        to_refresh = existing_sorted
    else:
        n = max(0, int(config.refresh_tail_days))
        to_refresh = existing_sorted[:n]
    for d in to_refresh:
        ordered.append(d)

    # Deduplicate while preserving order (latest may also be missing/existing).
    seen: set[date] = set()
    out: list[date] = []
    for d in ordered:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _update_market_data_polygon_grouped(config: LoaderConfig) -> None:
    ensure_dirs(config.paths)
    api_key = require_polygon_api_key(config.repo_root)

    # Shared, per-host limiter applies across all Polygon requests (including retries).
    configure_host_rate_limit("api.polygon.io", calls_per_minute=int(config.calls_per_minute))

    cal = TradingCalendar("NYSE")
    now_utc = datetime.now(timezone.utc)
    today = calendar_local_date(now_utc)

    existing = set(config.paths.list_polygon_grouped_daily_partitions().keys())
    planned_dates = _plan_polygon_dates(config=config, cal=cal, today=today, existing_partitions=existing, now_utc=now_utc)
    if not planned_dates:
        print("[yellow]No trading dates planned[/yellow]")
        ensure_last_n_bars(config)
        newest = max(existing) if existing else None
        write_update_state(
            config.paths,
            vendor=str(config.ohlcv_vendor),
            started_at=now_utc,
            newest_partition=newest,
        )
        print(f"[green]Update stamp[/green] {config.paths.update_state_json}")
        return

    refresh_note = (
        "full_refresh" if config.full_refresh else f"refresh_tail_days={config.refresh_tail_days}"
    )
    print(
        f"[cyan]Planning[/cyan] {len(planned_dates)} trading-day fetches "
        f"(lookback_years={config.lookback_years}, {refresh_note}, "
        f"calls_per_minute={config.calls_per_minute})"
    )

    vendor = PolygonGroupedDailyVendor()
    import requests

    completed = 0
    fetched_paths: list[Path] = []
    dates_updated: list[str] = []
    dates_failed: list[str] = []
    dates_no_data: list[str] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=False,
        refresh_per_second=10,
    ) as progress:
        task_id = progress.add_task("Fetching Polygon grouped daily", total=len(planned_dates))

        for d in planned_dates:
            try:
                df = None
                for attempt in range(0, max(1, int(config.max_retries)) + 1):
                    try:
                        df = vendor.fetch_grouped_daily(
                            trading_date=d,
                            api_key=api_key,
                            adjusted=bool(config.polygon_adjusted),
                            include_otc=bool(config.polygon_include_otc),
                            timeout_seconds=float(config.timeout_seconds),
                        )
                        break
                    except requests.HTTPError as e:
                        resp = getattr(e, "response", None)
                        status = getattr(resp, "status_code", None)
                        if status == 429 and attempt < int(config.max_retries):
                            retry_after = None
                            if resp is not None:
                                retry_after = resp.headers.get("Retry-After")
                            try:
                                sleep_s = float(retry_after) if retry_after else _retry_sleep(attempt)
                            except Exception:
                                sleep_s = _retry_sleep(attempt)
                            time.sleep(max(0.0, float(sleep_s)))
                            continue
                        raise

                out_path = config.paths.polygon_grouped_daily_parquet(d)
                tmp_path = Path(str(out_path) + ".tmp")

                if df is None or df.empty:
                    # Don't create empty partitions; just record progress.
                    print(f"[dim]NO_DATA[/dim] {d.isoformat()}")
                    dates_no_data.append(d.isoformat())
                else:
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(tmp_path, index=False)
                    from .paths import atomic_replace

                    atomic_replace(tmp_path, out_path)
                    fetched_paths.append(out_path)
                    dates_updated.append(d.isoformat())
                    print(f"[green]UPDATED[/green] {d.isoformat()} (rows={len(df)})")

            except requests.HTTPError as e:
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                # Avoid logging the full URL (it can include apiKey=...).
                if status is None:
                    print(f"[red]FAILED[/red] {d.isoformat()} - HTTPError")
                else:
                    print(f"[red]FAILED[/red] {d.isoformat()} - HTTP {status}")
                dates_failed.append(d.isoformat())
                if config.fail_fast:
                    code = status if status is not None else "error"
                    raise RuntimeError(
                        f"Polygon grouped daily failed for {d.isoformat()} (HTTP {code})"
                    ) from None
            except Exception as e:  # noqa: BLE001
                print(f"[red]FAILED[/red] {d.isoformat()} - {type(e).__name__}: {e}")
                dates_failed.append(d.isoformat())
                if config.fail_fast:
                    raise

            completed += 1
            progress.update(task_id, completed=completed)

    # Derived rebuild will be wired in a later step (derived-incremental).
    print("[cyan]Refreshing derived last-N cache[/cyan]")
    ensure_last_n_bars(config)
    newest = max(config.paths.list_polygon_grouped_daily_partitions().keys(), default=None)
    stamp = write_update_state(
        config.paths,
        vendor=str(config.ohlcv_vendor),
        started_at=now_utc,
        newest_partition=newest,
        dates_updated=dates_updated,
        dates_failed=dates_failed,
        dates_no_data=dates_no_data,
        ok=True,
    )
    print(
        f"[green]Update stamp[/green] {stamp.path} "
        f"(finished_at={stamp.finished_at.isoformat()}, newest={stamp.newest_partition})"
    )


def update_market_data(config: LoaderConfig) -> None:
    # Default to Polygon grouped-daily date partition mode.
    vendor_name = (config.ohlcv_vendor or "").strip().lower()
    if vendor_name in {"polygon", "polygon_grouped", "polygon_grouped_daily", "massive"}:
        return _update_market_data_polygon_grouped(config)
    return _update_market_data_per_ticker(config)

