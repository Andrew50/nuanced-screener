from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def window_embeddings_dir(self) -> Path:
        """
        Stack 4: on-disk embedding index for sliding windows over full history.

        Subdirs are expected:
        - fullinfo/   (no masking; offline historical mining)
        - openonly/   (mask last bar to open-only; screen-at-open semantics)
        """
        return self.derived_dir / "window_embeddings"

    @property
    def window_embeddings_fullinfo_dir(self) -> Path:
        return self.window_embeddings_dir / "fullinfo"

    @property
    def window_embeddings_openonly_dir(self) -> Path:
        return self.window_embeddings_dir / "openonly"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def patterns_dir(self) -> Path:
        """
        Stack 4: per-pattern artifacts (examples/confusers/prototypes/thresholds).
        """
        return self.data_dir / "patterns"

    @property
    def raw_by_date_dir(self) -> Path:
        return self.data_dir / "raw_by_date"

    @property
    def polygon_grouped_daily_dir(self) -> Path:
        return self.raw_by_date_dir / "polygon" / "grouped_daily"

    @property
    def logs_dir(self) -> Path:
        return self.meta_dir / "logs"

    @property
    def tickers_csv(self) -> Path:
        return self.meta_dir / "tickers.csv"

    @property
    def tickers_meta_json(self) -> Path:
        return self.meta_dir / "tickers.meta.json"

    @property
    def ticker_state_parquet(self) -> Path:
        return self.meta_dir / "ticker_state.parquet"

    @property
    def update_state_json(self) -> Path:
        """Atomic stamp written at the end of a successful `ns update` run."""
        return self.meta_dir / "update_state.json"

    @property
    def symbol_map_csv(self) -> Path:
        return self.meta_dir / "symbol_map.csv"

    @property
    def setups_dir(self) -> Path:
        return self.data_dir / "setups"

    @property
    def labels_parquet(self) -> Path:
        return self.labels_dir / "labels.parquet"

    @property
    def last_100_bars_parquet(self) -> Path:
        return self.derived_dir / "last_100_bars.parquet"

    @property
    def last_100_bars_manifest_json(self) -> Path:
        return self.derived_dir / "last_100_bars.manifest.json"

    @property
    def windowed_bars_parquet(self) -> Path:
        return self.derived_dir / "windowed_bars.parquet"

    def raw_ticker_parquet(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return self.raw_dir / f"{safe}.parquet"

    def polygon_grouped_daily_parquet(self, trading_date: date) -> Path:
        # Date-partitioned file naming (not hive-style directories).
        return self.polygon_grouped_daily_dir / f"date={trading_date.isoformat()}.parquet"

    def list_polygon_grouped_daily_partitions(self) -> dict[date, Path]:
        out: dict[date, Path] = {}
        if not self.polygon_grouped_daily_dir.exists():
            return out
        for p in self.polygon_grouped_daily_dir.glob("date=*.parquet"):
            name = p.name
            # date=YYYY-MM-DD.parquet
            try:
                d_str = name[len("date=") : -len(".parquet")]
                d = date.fromisoformat(d_str)
            except Exception:
                continue
            out[d] = p
        return out


def ensure_dirs(paths: DataPaths) -> None:
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    paths.setups_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_by_date_dir.mkdir(parents=True, exist_ok=True)
    paths.polygon_grouped_daily_dir.mkdir(parents=True, exist_ok=True)
    paths.derived_dir.mkdir(parents=True, exist_ok=True)
    paths.window_embeddings_dir.mkdir(parents=True, exist_ok=True)
    paths.window_embeddings_fullinfo_dir.mkdir(parents=True, exist_ok=True)
    paths.window_embeddings_openonly_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.patterns_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def atomic_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)

