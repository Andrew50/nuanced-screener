from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from screener_loader.config import LoaderConfig
from screener_loader.derived import (
    ensure_last_n_bars,
    last_n_cache_status,
    rebuild_last_n_bars_from_polygon_date_partitions,
)
from screener_loader.paths import ensure_dirs


def _write_partition(root: Path, d: date, close: float) -> None:
    cfg = LoaderConfig(repo_root=root)
    ensure_dirs(cfg.paths)
    out = cfg.paths.polygon_grouped_daily_parquet(d)
    pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": d,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100,
                "adj_close": None,
            }
        ]
    ).to_parquet(out, index=False)


def test_ensure_last_n_skips_when_inputs_unchanged(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=2, feature_columns=("ret_1d",))
    ensure_dirs(cfg.paths)
    _write_partition(tmp_path, date(2026, 1, 1), 10.0)
    _write_partition(tmp_path, date(2026, 1, 2), 11.0)
    first = ensure_last_n_bars(cfg)
    assert first.exists()
    assert cfg.paths.last_100_bars_manifest_json.exists()
    mtime = first.stat().st_mtime_ns
    status = last_n_cache_status(cfg)
    assert status.needs_rebuild is False
    again = ensure_last_n_bars(cfg)
    assert again.stat().st_mtime_ns == mtime


def test_ensure_last_n_rebuilds_when_partition_rewritten(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=2, feature_columns=("ret_1d",))
    ensure_dirs(cfg.paths)
    _write_partition(tmp_path, date(2026, 1, 1), 10.0)
    _write_partition(tmp_path, date(2026, 1, 2), 11.0)
    ensure_last_n_bars(cfg)
    _write_partition(tmp_path, date(2026, 1, 2), 12.0)
    status = last_n_cache_status(cfg)
    assert status.needs_rebuild is True
    assert "source" in status.reason
    ensure_last_n_bars(cfg)
    assert last_n_cache_status(cfg).needs_rebuild is False


def test_ensure_last_n_rebuilds_when_window_size_changes(tmp_path: Path) -> None:
    cfg = LoaderConfig(repo_root=tmp_path, window_size=2, feature_columns=("ret_1d",))
    ensure_dirs(cfg.paths)
    _write_partition(tmp_path, date(2026, 1, 1), 10.0)
    _write_partition(tmp_path, date(2026, 1, 2), 11.0)
    _write_partition(tmp_path, date(2026, 1, 3), 12.0)
    rebuild_last_n_bars_from_polygon_date_partitions(cfg)
    wider = LoaderConfig(repo_root=tmp_path, window_size=3, feature_columns=("ret_1d",))
    status = last_n_cache_status(wider)
    assert status.needs_rebuild is True
    assert "window_size" in status.reason


def test_update_owns_rebuild_screen_does_not() -> None:
    import inspect

    from screener_loader import update
    from screener_loader.cli import vision

    update_src = inspect.getsource(update)
    vision_src = inspect.getsource(vision)
    assert "ensure_last_n_bars" in update_src
    assert "rebuild_last_n_bars(" not in update_src
    assert "rebuild_last_n_bars_from_polygon" not in update_src
    assert "ensure_last_n" not in vision_src
    assert "rebuild_last_n" not in vision_src
    assert "update_market_data" in vision_src
