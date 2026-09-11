"""Vision scan CLI. Optional extras are imported lazily inside commands."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Optional
import csv
import json
import os
import subprocess
import sys

import typer
from rich import print
from rich.table import Table

from .common import _config
from .root import app
from ..update import update_market_data
from ..update_state import DEFAULT_STALE_AFTER, ensure_fresh_market_data, is_update_fresh, load_update_state

vision_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(vision_app, name="vision")


def _fail(exc: Exception) -> None:
    from screener_loader.setups.spec import MarketCapUnavailableError
    from screener_loader.vision.types import (
        ChartProfileConflictError,
        FrozenInputConflictError,
        InsufficientLastNError,
        StaleSnapshotError,
        VisionError,
    )

    if isinstance(
        exc,
        (
            VisionError,
            ChartProfileConflictError,
            StaleSnapshotError,
            InsufficientLastNError,
            FrozenInputConflictError,
            MarketCapUnavailableError,
            FileNotFoundError,
        ),
    ):
        hint = getattr(exc, "rebuild_hint", None)
        if hint:
            raise typer.BadParameter(f"{exc} Rebuild with `{hint}`.") from exc
        if isinstance(exc, InsufficientLastNError):
            raise typer.BadParameter(str(exc)) from exc
        raise typer.BadParameter(str(exc)) from exc
    raise exc


def _scan_config(
    *,
    model: str,
    mode: str,
    timeout_seconds: float,
    batch_size: int,
    max_output_tokens: int,
) -> object:
    from screener_loader.vision.types import ScanConfig

    return ScanConfig(
        model=model,
        mode=mode,  # type: ignore[arg-type]
        timeout_seconds=float(timeout_seconds),
        batch_size=int(batch_size),
        max_output_tokens=int(max_output_tokens),
    )


def _maybe_refresh_market_data(
    cfg,
    *,
    skip_update: bool,
    mode: str,
    stale_after_hours: float,
) -> None:
    """Run ``ns update`` when the stamp is stale. Last-N rebuild is update's job, not screen's."""

    if skip_update or str(mode) != "live":
        return
    max_age = timedelta(hours=float(stale_after_hours))
    state = load_update_state(cfg.paths)
    if is_update_fresh(state, max_age=max_age):
        assert state is not None
        print(
            f"[green]Data is fresh[/green] last update {state.finished_at.isoformat()} "
            f"(newest={state.newest_partition})"
        )
        return
    reason = (
        "no update stamp"
        if state is None
        else f"last update {state.finished_at.isoformat()} is older than {stale_after_hours:g}h"
    )
    print(f"[yellow]Data is stale[/yellow] ({reason}); running update")
    ensure_fresh_market_data(cfg, updater=update_market_data, max_age=max_age)


def _run_vision_screen(
    *,
    repo_root: Path,
    model: str | None,
    mode: str,
    dry_run: bool,
    demo: bool,
    max_candidates: int | None,
    timeout_seconds: float,
    batch_size: int,
    max_output_tokens: int,
    skip_update: bool,
    stale_after_hours: float,
) -> None:
    chosen = str(mode).strip().lower()
    if dry_run and demo:
        raise typer.BadParameter("Use only one of --dry-run or --demo")
    if dry_run:
        chosen = "dry_run"
    if demo:
        chosen = "demo"
    if chosen not in {"live", "dry_run", "demo"}:
        raise typer.BadParameter("mode must be live, dry_run, or demo")

    model_id = (model or "").strip()
    if not model_id:
        if chosen == "dry_run":
            model_id = "dry-run-model"
        elif chosen == "demo":
            model_id = "demo-vision-classifier"
        else:
            raise typer.BadParameter("Live scans require --model or NS_VISION_MODEL")

    from screener_loader.vision.service import VisionApp

    cfg = _config(repo_root=repo_root)
    _maybe_refresh_market_data(
        cfg,
        skip_update=skip_update,
        mode=chosen,
        stale_after_hours=stale_after_hours,
    )
    services = VisionApp(cfg, max_candidates=max_candidates)
    scan_config = _scan_config(
        model=model_id,
        mode=chosen,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )
    try:
        outcome = services.run(scan_config)
    except Exception as exc:
        _fail(exc)
        return
    print(
        f"[green]{outcome.status}[/green] run_id={outcome.run_id} "
        f"candidates={outcome.summary.candidates_total} "
        f"completed={outcome.summary.candidates_completed} "
        f"skipped={outcome.summary.candidates_skipped} "
        f"error={outcome.summary.candidates_error} "
        f"matches={outcome.summary.setup_matches} "
        f"synthetic={outcome.summary.synthetic}"
    )
    print(f"scan root: {services.scan_root}")


@app.command("screen")
def screen(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    model: Optional[str] = typer.Option(None, "--model", envvar="NS_VISION_MODEL"),
    mode: str = typer.Option("live", "--mode", help="live | dry_run | demo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render/compile/estimate only. Zero provider calls."),
    demo: bool = typer.Option(False, "--demo", help="Explicit labeled local classifier. Not a missing-key fallback."),
    max_candidates: Optional[int] = typer.Option(
        None,
        "--max-candidates",
        help="Optional cap for smoke tests. Applied after eligibility, before freeze.",
    ),
    timeout_seconds: float = typer.Option(60.0, "--timeout-seconds"),
    batch_size: int = typer.Option(10, "--batch-size"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    skip_update: bool = typer.Option(
        False,
        "--skip-update",
        help="Do not auto-run `ns update` when last-N is older than --stale-after-hours (live only).",
    ),
    stale_after_hours: float = typer.Option(
        DEFAULT_STALE_AFTER.total_seconds() / 3600.0,
        "--stale-after-hours",
        help="Treat market data as stale if the update stamp is older than this.",
    ),
) -> None:
    """LLM chart-setup screen over last-N bars. Alias: `ns vision scan`."""

    _run_vision_screen(
        repo_root=repo_root,
        model=model,
        mode=mode,
        dry_run=dry_run,
        demo=demo,
        max_candidates=max_candidates,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        skip_update=skip_update,
        stale_after_hours=stale_after_hours,
    )


@vision_app.command("scan")
def vision_scan(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    model: Optional[str] = typer.Option(None, "--model", envvar="NS_VISION_MODEL"),
    mode: str = typer.Option("live", "--mode", help="live | dry_run | demo"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render/compile/estimate only. Zero provider calls."),
    demo: bool = typer.Option(False, "--demo", help="Explicit labeled local classifier. Not a missing-key fallback."),
    max_candidates: Optional[int] = typer.Option(
        None,
        "--max-candidates",
        help="Optional cap for smoke tests. Applied after eligibility, before freeze.",
    ),
    timeout_seconds: float = typer.Option(60.0, "--timeout-seconds"),
    batch_size: int = typer.Option(10, "--batch-size"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    skip_update: bool = typer.Option(
        False,
        "--skip-update",
        help="Do not auto-run `ns update` when last-N is older than --stale-after-hours (live only).",
    ),
    stale_after_hours: float = typer.Option(
        DEFAULT_STALE_AFTER.total_seconds() / 3600.0,
        "--stale-after-hours",
        help="Treat market data as stale if the update stamp is older than this.",
    ),
) -> None:
    """Alias for `ns screen` (LLM chart-setup pipeline)."""

    _run_vision_screen(
        repo_root=repo_root,
        model=model,
        mode=mode,
        dry_run=dry_run,
        demo=demo,
        max_candidates=max_candidates,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        skip_update=skip_update,
        stale_after_hours=stale_after_hours,
    )


@vision_app.command("resume")
def vision_resume(
    run_id: str = typer.Option(..., "--run-id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    from screener_loader.vision.service import VisionApp

    cfg = _config(repo_root=repo_root)
    services = VisionApp(cfg)
    try:
        outcome = services.resume(run_id)
    except Exception as exc:
        _fail(exc)
        return
    print(
        f"[green]{outcome.status}[/green] run_id={outcome.run_id} "
        f"completed={outcome.summary.candidates_completed} "
        f"pending={outcome.summary.candidates_pending}"
    )


@vision_app.command("runs")
def vision_runs(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
) -> None:
    from screener_loader.vision.service import VisionApp

    cfg = _config(repo_root=repo_root)
    services = VisionApp(cfg)
    runs = services.reader.list_runs()
    if not runs:
        print("[yellow]No vision runs[/yellow]")
        return
    table = Table("run_id", "status", "created_at", "model", "synthetic")
    for run in runs:
        table.add_row(
            run.run_id,
            run.status,
            run.created_at.isoformat(),
            run.config.model,
            "yes" if run.synthetic else "no",
        )
    print(table)


@vision_app.command("export")
def vision_export(
    run_id: str = typer.Option(..., "--run-id"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    view: str = typer.Option("all", "--view", help="all | matches | no_matches | uncertain | errors"),
) -> None:
    from screener_loader.vision.service import VisionApp
    from screener_loader.vision.types import ResultQuery
    from screener_loader.vision_ui.components import view_query_parts

    cfg = _config(repo_root=repo_root)
    services = VisionApp(cfg)
    verdicts, statuses = view_query_parts(view if view != "all" else "all")
    query = ResultQuery(run_id=run_id, verdicts=verdicts)
    try:
        payload = services.reader.export_query(query, statuses=statuses)
        manifest = services.store.export_manifest(run_id)
    except Exception as exc:
        _fail(exc)
        return
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _write_dicts(dest / "candidates.csv", payload.candidates)
    _write_dicts(dest / "assessments.csv", payload.assessments)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[green]Wrote[/green] {dest} ({len(payload.candidates)} candidates, full filter not just the page)")


@vision_app.command("evaluate")
def vision_evaluate(
    run_id: str = typer.Option(..., "--run-id"),
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    from screener_loader.vision.evaluate import evaluate_from_stores
    from screener_loader.vision.service import VisionApp

    cfg = _config(repo_root=repo_root)
    services = VisionApp(cfg)
    try:
        report = evaluate_from_stores(run_id, store=services.store, reviews=services.reviews)
    except Exception as exc:
        _fail(exc)
        return
    if json_out:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    print(
        f"run={report.run_id} labeled={report.labeled_pairs}/{report.eligible_pairs} "
        f"coverage={report.coverage}"
    )
    table = Table("setup", "tp", "fp", "fn", "tn", "precision", "recall")
    for row in report.by_setup:
        table.add_row(
            row.setup_id,
            str(row.true_positive),
            str(row.false_positive),
            str(row.false_negative),
            str(row.true_negative),
            "n/a" if row.precision is None else f"{row.precision:.3f}",
            "n/a" if row.recall is None else f"{row.recall:.3f}",
        )
    print(table)
    for note in report.notes:
        print(f"[dim]{note}[/dim]")


@vision_app.command("view")
def vision_view(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    port: int = typer.Option(8501, "--port"),
    run_id: Optional[str] = typer.Option(None, "--run"),
) -> None:
    """Open the shared Streamlit app on the results page."""

    _launch_ui(repo_root=repo_root, port=port, page="results", run_id=run_id)


def launch_shared_ui(
    *,
    repo_root: Path,
    port: int,
    page: str = "builder",
    run_id: str | None = None,
) -> None:
    _launch_ui(repo_root=repo_root, port=port, page=page, run_id=run_id)


def _launch_ui(*, repo_root: Path, port: int, page: str, run_id: str | None) -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise typer.BadParameter("Streamlit is not installed. Run: pip install -e '.[ui]'") from e

    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    env = os.environ.copy()
    env["NS_REPO_ROOT"] = str(Path(repo_root).resolve())
    env["NS_VISION_PAGE"] = page
    if run_id:
        env["NS_VISION_RUN_ID"] = run_id
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd, env=env))


def _write_dicts(path: Path, rows: tuple[dict, ...]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
