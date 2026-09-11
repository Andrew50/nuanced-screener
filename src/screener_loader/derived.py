from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from rich import print

from .config import LoaderConfig
from .duckdb_utils import connect
from .feature_sql import (
    DAILY_RANGE_PCT_SQL,
    feature_sql_fragments,
    max_feature_lookback_days,
    merge_filter_features,
)
from .paths import atomic_replace, ensure_dirs

_LAST_N_MANIFEST_SCHEMA = 1
_POLYGON_VENDORS = {"polygon", "polygon_grouped", "polygon_grouped_daily", "massive"}


@dataclass(frozen=True)
class LastNCacheStatus:
    needs_rebuild: bool
    reason: str


def _resolved_feature_columns(config: LoaderConfig) -> tuple[str, ...]:
    return merge_filter_features(config.feature_columns)


def _feature_sql_clause(feature_columns: tuple[str, ...]) -> str:
    exprs = feature_sql_fragments(feature_columns)
    if not exprs:
        return ""
    return ",\n            " + ",\n            ".join(exprs)


def _empty_typed_nulls(feature_columns: tuple[str, ...]) -> list[str]:
    typed_nulls = [
        "CAST(NULL AS VARCHAR) AS ticker",
        "CAST(NULL AS DATE) AS date",
        "CAST(NULL AS DOUBLE) AS open",
        "CAST(NULL AS DOUBLE) AS high",
        "CAST(NULL AS DOUBLE) AS low",
        "CAST(NULL AS DOUBLE) AS close",
        "CAST(NULL AS BIGINT) AS volume",
        "CAST(NULL AS DOUBLE) AS adj_close",
    ]
    for c in feature_columns:
        typed_nulls.append(f"CAST(NULL AS DOUBLE) AS {c}")
    typed_nulls.append("CAST(NULL AS BIGINT) AS rn")
    return typed_nulls


def _copy_last_n_sql(source_rel: str, *, window_size: int, feature_columns: tuple[str, ...]) -> str:
    feature_sql = _feature_sql_clause(feature_columns)
    return f"""
        COPY (
          WITH src AS (
            SELECT
              ticker,
              CAST(date AS DATE) AS date,
              open,
              high,
              low,
              close,
              volume,
              adj_close
            FROM {source_rel}
          ),
          staged AS (
            SELECT
              *,
              {DAILY_RANGE_PCT_SQL}
            FROM src
          ),
          base AS (
            SELECT
              ticker,
              date,
              open,
              high,
              low,
              close,
              volume,
              adj_close
              {feature_sql}
            FROM staged
          ),
          ranked AS (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM base
          )
          SELECT *
          FROM ranked
          WHERE rn <= {int(window_size)}
        )
    """


def _sql_quote_path(p: Path) -> str:
    # DuckDB SQL single-quoted literal
    return "'" + str(p).replace("'", "''") + "'"


def uses_polygon_date_partitions(config: LoaderConfig) -> bool:
    return (config.ohlcv_vendor or "").strip().lower() in _POLYGON_VENDORS


def polygon_last_n_source_files(config: LoaderConfig) -> list[Path]:
    parts = config.paths.list_polygon_grouped_daily_partitions()
    if not parts:
        return []
    dates_sorted = sorted(parts.keys())
    feature_columns = _resolved_feature_columns(config)
    lookback = max_feature_lookback_days(feature_columns)
    k = int(config.window_size) + int(lookback) + 2
    if k <= 0:
        k = 1
    recent_dates = dates_sorted[-k:]
    return [parts[d] for d in recent_dates if d in parts]


def last_n_input_files(config: LoaderConfig) -> list[Path]:
    if uses_polygon_date_partitions(config):
        return polygon_last_n_source_files(config)
    return sorted(p for p in config.paths.raw_dir.glob("*.parquet") if p.is_file())


def _input_fingerprint(path: Path, data_dir: Path) -> dict:
    st = path.stat()
    resolved = path.resolve()
    data_root = data_dir.resolve()
    try:
        rel = str(resolved.relative_to(data_root))
    except ValueError:
        rel = path.name
    return {"path": rel, "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _last_n_manifest_payload(config: LoaderConfig, input_files: list[Path]) -> dict:
    data_dir = config.paths.data_dir
    files = sorted(input_files, key=lambda p: str(p))
    return {
        "schema_version": _LAST_N_MANIFEST_SCHEMA,
        "window_size": int(config.window_size),
        "feature_columns": list(_resolved_feature_columns(config)),
        "vendor": str(config.ohlcv_vendor or ""),
        "inputs": [_input_fingerprint(p, data_dir) for p in files],
    }


def write_last_n_manifest(config: LoaderConfig, input_files: list[Path] | None = None) -> Path:
    files = list(input_files) if input_files is not None else last_n_input_files(config)
    path = config.paths.last_100_bars_manifest_json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_last_n_manifest_payload(config, files), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)
    return path


def last_n_cache_status(config: LoaderConfig) -> LastNCacheStatus:
    """Whether ``last_100_bars.parquet`` is still a valid projection of current inputs."""

    out = config.paths.last_100_bars_parquet
    if not out.exists():
        return LastNCacheStatus(True, "last-N file is missing")
    manifest_path = config.paths.last_100_bars_manifest_json
    if not manifest_path.exists():
        return LastNCacheStatus(True, "last-N manifest is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LastNCacheStatus(True, "last-N manifest is unreadable")
    if not isinstance(payload, dict):
        return LastNCacheStatus(True, "last-N manifest is invalid")
    if int(payload.get("schema_version") or 0) != _LAST_N_MANIFEST_SCHEMA:
        return LastNCacheStatus(True, "last-N manifest schema changed")
    if int(payload.get("window_size") or 0) != int(config.window_size):
        return LastNCacheStatus(True, "window_size changed")
    if str(payload.get("vendor") or "") != str(config.ohlcv_vendor or ""):
        return LastNCacheStatus(True, "vendor changed")
    stored_features = tuple(str(x) for x in (payload.get("feature_columns") or []))
    if stored_features != _resolved_feature_columns(config):
        return LastNCacheStatus(True, "feature columns changed")
    expected = _last_n_manifest_payload(config, last_n_input_files(config))["inputs"]
    if list(payload.get("inputs") or []) != expected:
        return LastNCacheStatus(True, "source partitions changed")
    return LastNCacheStatus(False, "current")


def last_n_needs_rebuild(config: LoaderConfig) -> bool:
    return last_n_cache_status(config).needs_rebuild


def rebuild_last_n_for_config(config: LoaderConfig) -> Path:
    if uses_polygon_date_partitions(config):
        return rebuild_last_n_bars_from_polygon_date_partitions(config)
    return rebuild_last_n_bars(config)


def ensure_last_n_bars(config: LoaderConfig) -> Path:
    """Rebuild last-N only when the derived cache is missing or its inputs changed.

    ``ns screen`` does not call this; it runs ``ns update`` on a stale stamp, and update
    calls this after fetch.
    """

    status = last_n_cache_status(config)
    if not status.needs_rebuild:
        print(f"[dim]Derived last-N is current; skipping rebuild[/dim]")
        return config.paths.last_100_bars_parquet
    print(f"[cyan]Last-N cache stale[/cyan] ({status.reason}); rebuilding")
    return rebuild_last_n_for_config(config)


def rebuild_last_n_bars_from_files(config: LoaderConfig, parquet_files: list[Path]) -> Path:
    """
    Build last-N bars from an explicit list of Parquet files (e.g. date partitions).
    """
    ensure_dirs(config.paths)
    out_path = config.paths.last_100_bars_parquet
    tmp_path = Path(str(out_path) + ".tmp")

    con = connect(config)

    feature_columns = _resolved_feature_columns(config)
    if not parquet_files:
        # Create an empty Parquet with a stable schema so downstream queries fail less often.
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        typed_nulls = _empty_typed_nulls(feature_columns)

        con.execute(
            f"""
            COPY (
              SELECT
                {", ".join(typed_nulls)}
              WHERE FALSE
            )
            TO '{tmp_path.as_posix()}'
            (FORMAT PARQUET, CODEC 'ZSTD');
            """
        )
        atomic_replace(tmp_path, out_path)
        write_last_n_manifest(config, parquet_files)
        print(f"[yellow]Derived[/yellow] wrote empty {out_path}")
        return out_path

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    files_sql = "[" + ", ".join(_sql_quote_path(p) for p in parquet_files) + "]"
    sql = _copy_last_n_sql(f"read_parquet({files_sql})", window_size=window_size, feature_columns=feature_columns)
    con.execute(
        f"""
        {sql}
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, out_path)
    write_last_n_manifest(config, parquet_files)
    print(f"[green]Derived[/green] wrote {out_path}")
    return out_path


def rebuild_last_n_bars_from_polygon_date_partitions(config: LoaderConfig) -> Path:
    """
    Efficient derived rebuild for Polygon date-partitioned raw data:
    read only the most recent K partitions where K ~= window_size + max feature lookback.
    """
    return rebuild_last_n_bars_from_files(config, polygon_last_n_source_files(config))


def rebuild_last_n_bars(config: LoaderConfig) -> Path:
    """
    Rebuild consolidated derived dataset containing last `window_size` bars per ticker.
    This is the primary screener input for market-wide scans.
    """
    ensure_dirs(config.paths)
    raw_glob = (config.paths.raw_dir / "*.parquet").as_posix()
    out_path = config.paths.last_100_bars_parquet
    tmp_path = Path(str(out_path) + ".tmp")

    con = connect(config)
    feature_columns = _resolved_feature_columns(config)
    raw_files = list(config.paths.raw_dir.glob("*.parquet"))
    if not raw_files:
        window_size = int(config.window_size)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        typed_nulls = _empty_typed_nulls(feature_columns)
        con.execute(
            f"""
            COPY (
              SELECT
                {", ".join(typed_nulls)}
              WHERE FALSE
            )
            TO '{tmp_path.as_posix()}'
            (FORMAT PARQUET, CODEC 'ZSTD');
            """
        )
        atomic_replace(tmp_path, out_path)
        write_last_n_manifest(config, raw_files)
        print(f"[yellow]Derived[/yellow] no raw files; wrote empty {out_path}")
        return out_path

    window_size = int(config.window_size)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    sql = _copy_last_n_sql(f"read_parquet('{raw_glob}')", window_size=window_size, feature_columns=feature_columns)
    con.execute(
        f"""
        {sql}
        TO '{tmp_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'ZSTD');
        """
    )
    atomic_replace(tmp_path, out_path)
    write_last_n_manifest(config, raw_files)
    print(f"[green]Derived[/green] wrote {out_path}")
    return out_path

