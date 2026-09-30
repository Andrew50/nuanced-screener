"""ScanSource: SetupService + eligibility + universe ∩ last-N + snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ...config import LoaderConfig
from ...setups.eligibility import compute_eligibility
from ...setups.service import SetupService
from ...setups.spec import effective_min
from ...setups import store as setup_store
from ...universe import load_universe
from ..serialization import digest_bytes
from ..snapshots import (
    assert_market_cap_unused_for_scan,
    feature_value_from_mapping,
    freeze_prepared_scan,
    make_candidate_input,
    resolve_chart_profile,
    snapshot_setup,
)
from ..types import (
    FeatureValue,
    InputSkip,
    PreparedScan,
    ScanConfig,
    ScanDiagnostics,
    StaleSnapshotError,
    VisionError,
)
from .bars import (
    inspect_last_n_capacity,
    load_candidate_windows,
    load_example_input,
    raise_insufficient_last_n,
    rebuild_last100_hint,
)
from .freshness import (
    as_utc,
    authorized_daily_session,
    coerce_date,
    expected_completed_session,
    resolve_scan_session,
    utc_now,
)

Clock = Callable[[], datetime]


def vision_scan_root(loader: LoaderConfig) -> Path:
    """Durable scan root under the repo data dir. Does not edit paths.py."""

    return loader.paths.data_dir / "vision_scans"


@dataclass(frozen=True)
class SourceRevision:
    tokens: tuple[tuple[str, str], ...]

    def mismatch_message(self, other: "SourceRevision") -> str | None:
        ours = dict(self.tokens)
        theirs = dict(other.tokens)
        keys = sorted(set(ours) | set(theirs))
        changed = [k for k in keys if ours.get(k) != theirs.get(k)]
        if not changed:
            return None
        return "Source changed during prepare: " + ", ".join(changed[:12])


def normalize_screen_tickers(values: Sequence[str] | None) -> tuple[str, ...]:
    """Uppercase, de-dupe, and split comma-separated ticker lists. Empty means the whole universe."""

    if not values:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in str(raw).replace(";", ",").split(","):
            ticker = part.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            out.append(ticker)
    return tuple(out)


def tickers_for_window_load(
    members: Sequence[str],
    max_candidates: int | None,
    *,
    oversample: int = 4,
) -> tuple[str, ...]:
    """Stable ticker order for last-N loads. A positive cap avoids scanning the whole universe."""

    ordered = tuple(sorted({str(t) for t in members}))
    if max_candidates is None:
        return ordered
    cap = int(max_candidates)
    if cap < 0:
        raise VisionError("max_candidates must be >= 0")
    if cap == 0:
        return ordered
    return ordered[: min(len(ordered), max(cap * oversample, cap))]


class SetupScanSource:
    """Bind existing catalog, eligibility, last-N bars, and tickers.csv."""

    def __init__(
        self,
        loader: LoaderConfig,
        *,
        clock: Clock | None = None,
        max_candidates: int | None = None,
        tickers: Sequence[str] | None = None,
        calendar_name: str = "NYSE",
    ) -> None:
        self.loader = loader
        self._clock = clock or utc_now
        self.max_candidates = max_candidates
        self.tickers = normalize_screen_tickers(tickers)
        self.calendar_name = calendar_name

    def prepare(self, config: ScanConfig) -> PreparedScan:
        loader = self.loader
        service = SetupService(loader.paths, market_cap_available=False)
        revision = capture_source_revision(loader)

        enabled = [service.get(spec.id) for spec in service.list_enabled()]
        if not enabled:
            raise VisionError("No enabled setups to scan")
        assert_market_cap_unused_for_scan(enabled, market_cap_available=False)
        profile = resolve_chart_profile(enabled)

        last_n = loader.paths.last_100_bars_parquet
        if not last_n.exists():
            raise FileNotFoundError(
                f"Derived last-N bars not found: {last_n}. "
                f"Run `{rebuild_last100_hint(profile.lookback_bars, loader.repo_root)}` first."
            )

        now = as_utc(self._clock())
        completed = expected_completed_session(now, calendar_name=self.calendar_name)
        authorized = authorized_daily_session(now, calendar_name=self.calendar_name)
        capacity = inspect_last_n_capacity(loader)
        if capacity.max_rn < int(profile.lookback_bars):
            raise_insufficient_last_n(profile.lookback_bars, loader.repo_root)
        try:
            expected = resolve_scan_session(
                mode=config.mode,
                expected=completed,
                authorized=authorized,
                max_row_date=capacity.max_date,
            )
        except StaleSnapshotError as exc:
            exc.rebuild_hint = (  # type: ignore[attr-defined]
                f"ns update --repo-root {loader.repo_root.resolve()} && "
                f"{rebuild_last100_hint(profile.lookback_bars, loader.repo_root)}"
            )
            raise

        universe = load_universe(loader)
        universe_tickers = {str(t) for t in universe["ticker"].astype(str).tolist()}
        eligibility = compute_eligibility(loader, service, setups=enabled)

        later = capture_source_revision(loader)
        mismatch = revision.mismatch_message(later)
        if mismatch:
            raise VisionError(mismatch)

        members: dict[str, tuple[str, ...]] = {}
        for ticker, setup_ids in eligibility.eligible_setups.items():
            key = str(ticker)
            if key in universe_tickers:
                members[key] = tuple(setup_ids)

        exclusion: tuple[InputSkip, ...] = ()
        if self.tickers:
            missing = tuple(t for t in self.tickers if t not in members)
            exclusion = _exclusion_skips(
                loader,
                missing,
                universe_tickers=universe_tickers,
                global_filters=service.load_global_filters(),
                setups=enabled,
            )
            members = {t: members[t] for t in self.tickers if t in members}

        per_setup = Counter()
        for setup_ids in members.values():
            for sid in setup_ids:
                per_setup[sid] += 1

        feature_asof, features = _features_from_eligibility(eligibility.tickers, members)
        window_tickers = tickers_for_window_load(tuple(members), self.max_candidates)
        windows, skips = load_candidate_windows(
            loader,
            tickers=window_tickers,
            lookback_bars=int(profile.lookback_bars),
            expected_session=expected,
            feature_asof=feature_asof,
            volume_required=bool(profile.volume),
        )
        skipped_tickers = {s.ticker for s in skips}
        skips = tuple(replace(s, features=features.get(s.ticker, s.features)) for s in skips)
        skips = exclusion + skips
        candidates = []
        for ticker in sorted(windows):
            if ticker in skipped_tickers:
                continue
            candidates.append(
                make_candidate_input(
                    ticker=ticker,
                    window=windows[ticker],
                    features=features.get(ticker, FeatureValue(None, None, None)),
                    eligible_setup_ids=members[ticker],
                    profile=profile,
                )
            )
        if self.max_candidates is not None:
            cap = int(self.max_candidates)
            if cap < 0:
                raise VisionError("max_candidates must be >= 0")
            candidates = sorted(candidates, key=lambda c: c.candidate_id)[:cap]

        setups_out = []
        examples_out = []
        for spec in enabled:
            yaml_bytes = setup_store.setup_yaml_path(loader.paths, spec.id).read_bytes()
            catalog_examples = service.load_examples(spec.id)
            snap = snapshot_setup(spec, yaml_bytes=yaml_bytes, examples=catalog_examples)
            setups_out.append(snap)
            by_id = {ex.id: ex for ex in catalog_examples}
            for example_id in snap.example_ids:
                example = by_id[example_id]
                examples_out.append(
                    load_example_input(
                        loader,
                        spec,
                        example,
                        lookback_bars=int(profile.lookback_bars),
                    )
                )

        global_path = setup_store.global_yaml_path(loader.paths)
        global_bytes = global_path.read_bytes() if global_path.exists() else b""

        final = capture_source_revision(loader)
        mismatch = revision.mismatch_message(final)
        if mismatch:
            raise VisionError(mismatch)

        diagnostics = ScanDiagnostics(
            universe_count=int(len(universe_tickers)),
            input_count=int(capacity.ticker_count),
            eligible_union_count=int(len(members)),
            per_setup_eligible_counts=tuple(sorted(per_setup.items())),
            not_eligible_count=None,
            skipped_count=len(skips),
            unavailable=("not_eligible",),
        )
        return freeze_prepared_scan(
            profile=profile,
            setups=setups_out,
            examples=examples_out,
            candidates=tuple(candidates),
            config=config,
            diagnostics=diagnostics,
            skips=skips,
            global_filters_yaml_bytes=global_bytes,
            prepared_at=now,
        )


def _exclusion_skips(
    loader: LoaderConfig,
    tickers: Sequence[str],
    *,
    universe_tickers: set[str],
    global_filters,
    setups: Sequence,
) -> tuple[InputSkip, ...]:
    """Explain requested tickers that never became candidates. These are filter misses, not model verdicts."""

    if not tickers:
        return ()
    rows = _latest_feature_rows(loader, tickers)
    out: list[InputSkip] = []
    for ticker in tickers:
        row = rows.get(ticker)
        message = explain_ticker_exclusion(
            ticker,
            in_universe=ticker in universe_tickers,
            row=row,
            global_filters=global_filters,
            setups=setups,
        )
        asof = coerce_date(row.get("asof_date")) if row else None
        features = feature_value_from_mapping(row) if row else None
        out.append(
            InputSkip(
                ticker=ticker,
                kind="unavailable_input",
                message=message,
                asof_date=asof,
                features=features,
            )
        )
    return tuple(out)


def explain_ticker_exclusion(
    ticker: str,
    *,
    in_universe: bool,
    row: Mapping[str, object] | None,
    global_filters,
    setups: Sequence,
) -> str:
    if row is None and not in_universe:
        return f"{ticker} is not in the ticker universe and has no latest last-N bar"
    if not in_universe:
        return f"{ticker} is not in the ticker universe (tickers.csv)"
    if row is None:
        return f"{ticker} has no latest bar in last-N"
    close = _num(row.get("close"))
    dollar_vol = _num(row.get("dollar_vol_avg_20"))
    adr = _num(row.get("adr_pct_20"))
    global_bits = _threshold_bits(
        close=close,
        dollar_vol=dollar_vol,
        adr=adr,
        min_price=global_filters.min_price,
        min_dollar_vol=global_filters.min_dollar_vol_20d,
        min_adr=None,
    )
    if global_bits:
        return f"{ticker} did not pass screener filters: " + "; ".join(global_bits)
    setup_bits: list[str] = []
    for spec in setups:
        bits = _threshold_bits(
            close=close,
            dollar_vol=dollar_vol,
            adr=adr,
            min_price=effective_min(spec.filters.min_price, global_filters.min_price),
            min_dollar_vol=effective_min(spec.filters.min_dollar_vol_20d, global_filters.min_dollar_vol_20d),
            min_adr=spec.filters.min_adr_pct_20,
        )
        if bits:
            setup_bits.append(f"{spec.id}: " + ", ".join(bits))
    if setup_bits:
        return f"{ticker} did not pass screener filters: " + "; ".join(setup_bits)
    return f"{ticker} did not match any enabled setup"


def _threshold_bits(
    *,
    close: float | None,
    dollar_vol: float | None,
    adr: float | None,
    min_price: float | None,
    min_dollar_vol: float | None,
    min_adr: float | None,
) -> list[str]:
    bits: list[str] = []
    if min_price is not None and (close is None or close < float(min_price)):
        bits.append(f"close {_fmt(close)} < min_price {_fmt(min_price)}")
    if min_dollar_vol is not None and (dollar_vol is None or dollar_vol < float(min_dollar_vol)):
        bits.append(f"dollar_vol_avg_20 {_fmt(dollar_vol)} < min_dollar_vol_20d {_fmt(min_dollar_vol)}")
    if min_adr is not None and (adr is None or adr < float(min_adr)):
        bits.append(f"adr_pct_20 {_fmt(adr)} < min_adr_pct_20 {_fmt(min_adr)}")
    return bits


def _latest_feature_rows(loader: LoaderConfig, tickers: Sequence[str]) -> dict[str, dict]:
    from ...duckdb_utils import connect

    if not tickers:
        return {}
    placeholders = ", ".join("?" for _ in tickers)
    con = connect(loader)
    frame = con.execute(
        f"""
        SELECT
          UPPER(CAST(ticker AS VARCHAR)) AS ticker,
          CAST(date AS DATE) AS asof_date,
          CAST(close AS DOUBLE) AS close,
          CAST(dollar_vol_avg_20 AS DOUBLE) AS dollar_vol_avg_20,
          CAST(adr_pct_20 AS DOUBLE) AS adr_pct_20
        FROM read_parquet(?)
        WHERE rn = 1 AND UPPER(CAST(ticker AS VARCHAR)) IN ({placeholders})
        """,
        [str(loader.paths.last_100_bars_parquet), *tickers],
    ).df()
    out: dict[str, dict] = {}
    if frame is None or frame.empty:
        return out
    for _, row in frame.iterrows():
        out[str(row["ticker"])] = row.to_dict()
    return out


def _num(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def capture_source_revision(loader: LoaderConfig) -> SourceRevision:
    tokens: list[tuple[str, str]] = []
    paths = [loader.paths.last_100_bars_parquet, loader.paths.tickers_csv]
    global_yaml = setup_store.global_yaml_path(loader.paths)
    paths.append(global_yaml)
    if loader.paths.setups_dir.exists():
        for setup_id in setup_store.list_setup_ids(loader.paths):
            paths.append(setup_store.setup_yaml_path(loader.paths, setup_id))
            paths.append(setup_store.examples_yaml_path(loader.paths, setup_id))
            assets = setup_store.examples_assets_dir(loader.paths, setup_id)
            if assets.exists():
                for asset in sorted(assets.iterdir()):
                    if asset.is_file():
                        paths.append(asset)
    for path in paths:
        tokens.append((str(path), _path_token(path)))
    return SourceRevision(tokens=tuple(tokens))


def _path_token(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        data = path.read_bytes()
    except OSError:
        stat = path.stat()
        return f"meta:{stat.st_size}:{stat.st_mtime_ns}"
    if path.suffix.lower() == ".parquet" or path.stat().st_size > 1_000_000:
        stat = path.stat()
        return f"meta:{stat.st_size}:{stat.st_mtime_ns}"
    return digest_bytes(data)


def _features_from_eligibility(
    tickers_df,
    members: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, date], dict[str, FeatureValue]]:
    asof: dict[str, date] = {}
    features: dict[str, FeatureValue] = {}
    if tickers_df is None or getattr(tickers_df, "empty", True):
        return asof, features
    for _, row in tickers_df.iterrows():
        ticker = str(row["ticker"])
        if ticker not in members:
            continue
        d = coerce_date(row.get("asof_date", row.get("date")))
        if d is not None:
            asof[ticker] = d
        features[ticker] = feature_value_from_mapping(row)
    return asof, features
