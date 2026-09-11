from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import print

from ..derived import rebuild_last_n_bars_from_polygon_date_partitions
from ..screening import run_named_query
from ..universe import build_universe
from ..update import update_market_data
from ..update_state import is_update_fresh, load_update_state
from .common import _config
from .root import app


@app.command()
def universe(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ticker_source: str = typer.Option("nasdaq_trader", "--ticker-source"),
    exclude_test_issues: bool = typer.Option(True, "--exclude-test-issues/--include-test-issues"),
    exclude_etfs: bool = typer.Option(False, "--exclude-etfs/--include-etfs"),
    include_exchanges: tuple[str, str, str] = typer.Option(("NASDAQ", "NYSE", "AMEX"), "--include-exchanges"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ticker_source=ticker_source,
        exclude_test_issues=exclude_test_issues,
        exclude_etfs=exclude_etfs,
        include_exchanges=include_exchanges,
    )
    out = build_universe(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def update(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    ohlcv_vendor: str = typer.Option("polygon_grouped", "--ohlcv-vendor"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    full_refresh: bool = typer.Option(
        False,
        "--full-refresh",
        help="Polygon date-mode: also re-fetch every already-loaded day in the lookback window.",
    ),
    batch_size: int = typer.Option(50, "--batch-size", envvar="NS_BATCH_SIZE"),
    processes: int = typer.Option(8, "--processes", envvar="NS_PROCESSES"),
    executor: str = typer.Option("threads", "--executor", envvar="NS_EXECUTOR"),
    pause_seconds: float = typer.Option(0.0, "--pause-seconds", envvar="NS_PAUSE_SECONDS"),
    max_retries: int = typer.Option(3, "--max-retries", envvar="NS_MAX_RETRIES"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds", envvar="NS_TIMEOUT_SECONDS"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
    lookback_years: int = typer.Option(2, "--lookback-years", envvar="NS_LOOKBACK_YEARS", help="Polygon date-mode: years to backfill."),
    refresh_tail_days: int = typer.Option(
        3,
        "--refresh-tail-days",
        help="Polygon date-mode: re-fetch the N most recent already-loaded days (ignored with --full-refresh).",
    ),
    adjusted: bool = typer.Option(True, "--adjusted/--unadjusted", help="Polygon: adjusted prices."),
    include_otc: bool = typer.Option(False, "--include-otc/--exclude-otc", help="Polygon: include OTC tickers."),
    calls_per_minute: int = typer.Option(5, "--calls-per-minute", envvar="NS_CALLS_PER_MINUTE", help="Polygon free tier: max calls per minute."),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        ohlcv_vendor=ohlcv_vendor,
        start_date=start_date,
        end_date=end_date,
        full_refresh=full_refresh,
        batch_size=batch_size,
        processes=processes,
        executor=executor,
        pause_seconds=pause_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
        lookback_years=lookback_years,
        refresh_tail_days=refresh_tail_days,
        polygon_adjusted=adjusted,
        polygon_include_otc=include_otc,
        calls_per_minute=calls_per_minute,
    )
    update_market_data(cfg)
    stamp = load_update_state(cfg.paths)
    if stamp is not None:
        print(
            f"[green]Last update[/green] {stamp.finished_at.isoformat()} "
            f"(newest={stamp.newest_partition})"
        )


@app.command("update-status")
def update_status(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    stale_after_hours: float = typer.Option(24.0, "--stale-after-hours"),
) -> None:
    cfg = _config(repo_root=repo_root)
    state = load_update_state(cfg.paths)
    if state is None:
        print("[yellow]No update stamp[/yellow] (missing data/meta/update_state.json). Run `ns update`.")
        raise typer.Exit(code=1)
    max_age = timedelta(hours=float(stale_after_hours))
    fresh = is_update_fresh(state, max_age=max_age)
    age_hours = (datetime.now(timezone.utc) - state.finished_at).total_seconds() / 3600.0
    print(f"stamp:     {state.path}")
    print(f"finished:  {state.finished_at.isoformat()}")
    print(f"age_hours: {age_hours:.2f}")
    print(f"vendor:    {state.vendor}")
    print(f"newest:    {state.newest_partition}")
    print(f"updated:   {len(state.dates_updated)}")
    print(f"failed:    {len(state.dates_failed)}")
    print(f"fresh:     {fresh} (stale after {stale_after_hours:g}h)")
    if not fresh:
        raise typer.Exit(code=2)


@app.command("rebuild-last100")
def rebuild_last100(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    window_size: int = typer.Option(100, "--window-size"),
    feature_columns: list[str] = typer.Option([], "--feature-column"),
    duckdb_threads: int = typer.Option(4, "--duckdb-threads", envvar="NS_DUCKDB_THREADS"),
) -> None:
    cfg = _config(
        repo_root=repo_root,
        window_size=window_size,
        feature_columns=feature_columns,
        duckdb_threads=duckdb_threads,
    )
    out = rebuild_last_n_bars_from_polygon_date_partitions(cfg)
    print(f"[green]Wrote[/green] {out}")


@app.command()
def query(
    query: str = typer.Option(..., "--query", help="Named DuckDB query (see screener_loader/screening.py)."),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Run a named DuckDB query over last-N bars (SQL screen, not the LLM pipeline)."""
    cfg = _config(repo_root=repo_root)
    df = run_named_query(cfg, query=query, limit=limit)
    print(df)

