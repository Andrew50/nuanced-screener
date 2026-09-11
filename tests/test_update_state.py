from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from screener_loader.cli import app
from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.update_state import (
    DEFAULT_STALE_AFTER,
    ensure_fresh_market_data,
    is_update_fresh,
    load_update_state,
    write_update_state,
)


def _cfg(tmp_path: Path) -> LoaderConfig:
    cfg = LoaderConfig(repo_root=tmp_path)
    ensure_dirs(cfg.paths)
    return cfg


def test_write_and_load_update_state_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    started = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 9, 9, 12, 5, tzinfo=timezone.utc)
    written = write_update_state(
        cfg.paths,
        vendor="polygon_grouped",
        started_at=started,
        finished_at=finished,
        newest_partition=date(2026, 9, 8),
        dates_updated=["2026-09-08"],
        dates_failed=["2024-09-04"],
        ok=True,
    )
    assert written.path == cfg.paths.update_state_json
    assert written.path.exists()
    loaded = load_update_state(cfg.paths)
    assert loaded is not None
    assert loaded.finished_at == finished
    assert loaded.newest_partition == date(2026, 9, 8)
    assert loaded.dates_updated == ("2026-09-08",)
    assert loaded.dates_failed == ("2024-09-04",)
    assert loaded.vendor == "polygon_grouped"
    assert loaded.ok is True


def test_missing_or_corrupt_stamp_is_not_fresh(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert load_update_state(cfg.paths) is None
    assert is_update_fresh(None) is False
    cfg.paths.update_state_json.write_text("{not json", encoding="utf-8")
    assert load_update_state(cfg.paths) is None


def test_freshness_uses_finished_at_not_mtime(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    now = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)
    write_update_state(
        cfg.paths,
        vendor="polygon_grouped",
        finished_at=now - timedelta(hours=23),
        newest_partition=date(2026, 9, 8),
    )
    state = load_update_state(cfg.paths)
    assert is_update_fresh(state, now=now, max_age=DEFAULT_STALE_AFTER) is True
    assert is_update_fresh(state, now=now + timedelta(hours=2), max_age=DEFAULT_STALE_AFTER) is False


def test_failed_stamp_is_not_fresh(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    write_update_state(cfg.paths, vendor="polygon_grouped", ok=False)
    assert is_update_fresh(load_update_state(cfg.paths)) is False


def test_ensure_fresh_runs_updater_when_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    calls: list[int] = []

    def updater(_cfg: LoaderConfig) -> None:
        calls.append(1)
        write_update_state(_cfg.paths, vendor="polygon_grouped", newest_partition=date(2026, 9, 8))

    assert ensure_fresh_market_data(cfg, updater=updater) is True
    assert calls == [1]
    assert ensure_fresh_market_data(cfg, updater=updater) is False
    assert calls == [1]


def test_cli_update_status_and_screen_skip_update(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    write_update_state(
        cfg.paths,
        vendor="polygon_grouped",
        finished_at=datetime.now(timezone.utc),
        newest_partition=date(2026, 9, 8),
        dates_updated=["2026-09-08"],
    )
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": date(2026, 9, 8),
                "close": 10.0,
                "volume": 1000,
                "ret_21d": 0.2,
                "rn": 1,
            }
        ]
    ).to_parquet(cfg.paths.last_100_bars_parquet, index=False)

    runner = CliRunner()
    status = runner.invoke(app, ["update-status", "--repo-root", str(tmp_path)])
    assert status.exit_code == 0, status.output
    assert "fresh:     True" in status.output
    assert "2026-09-08" in status.output

    screen = runner.invoke(
        app,
        ["query", "--query", "top_momentum_21d", "--repo-root", str(tmp_path)],
    )
    assert screen.exit_code == 0, screen.output
    assert "AAA" in screen.output


def test_cli_query_does_not_auto_update(tmp_path: Path, monkeypatch) -> None:
    _cfg(tmp_path)
    calls: list[str] = []

    def fake_update(config: LoaderConfig) -> None:
        calls.append(str(config.repo_root))

    monkeypatch.setattr("screener_loader.cli.data.update_market_data", fake_update)
    pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "date": date(2026, 9, 8),
                "close": 11.0,
                "volume": 1000,
                "ret_21d": 0.3,
                "rn": 1,
            }
        ]
    ).to_parquet(LoaderConfig(repo_root=tmp_path).paths.last_100_bars_parquet, index=False)
    runner = CliRunner()
    res = runner.invoke(app, ["query", "--query", "top_momentum_21d", "--repo-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert calls == []
    assert "BBB" in res.output
