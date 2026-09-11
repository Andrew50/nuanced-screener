"""Bind existing setup catalog types to frozen vision snapshots."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from ..setups.prompt import compile_prompt
from ..setups.spec import (
    ChartStyle,
    MarketCapUnavailableError,
    SetupSpec,
    VisionExample,
)
from .serialization import digest, digest_bytes, json_number
from .types import (
    Bar,
    BarWindow,
    CandidateInput,
    ChartProfile,
    ChartProfileConflictError,
    ExampleInput,
    FeatureValue,
    FilterSnapshot,
    InputSkip,
    MaAvailability,
    PreparedScan,
    ProfileFieldConflict,
    ScanConfig,
    ScanDiagnostics,
    SetupSnapshot,
    SnapshotRule,
    VisionError,
)


def rule_id(setup_id: str, kind: str, index: int) -> str:
    return f"{setup_id}:{kind}:{int(index)}"


def scoped_example_id(setup_id: str, example_id: str) -> str:
    return f"{setup_id}/{example_id}"


def snapshot_rules(spec: SetupSpec) -> tuple[SnapshotRule, ...]:
    """Positional IDs scoped to this snapshot. Reordering criteria assigns new IDs."""
    out: list[SnapshotRule] = []
    for kind in ("required", "preferred", "disqualifiers"):
        key = "disqualifier" if kind == "disqualifiers" else kind
        texts = getattr(spec.criteria, kind)
        for i, text in enumerate(texts):
            out.append(
                SnapshotRule(
                    rule_id=rule_id(spec.id, key, i),
                    kind=key,  # type: ignore[arg-type]
                    index=i,
                    text=str(text),
                )
            )
    return tuple(out)


def ordered_example_ids(spec: SetupSpec, examples: Sequence[VisionExample]) -> tuple[str, ...]:
    """Public ordering only: polarity, quality, then id. Same as compile_prompt(...).example_ids."""
    return compile_prompt(spec, list(examples)).example_ids


def snapshot_setup(
    spec: SetupSpec,
    *,
    yaml_bytes: bytes,
    examples: Sequence[VisionExample],
) -> SetupSnapshot:
    rules = snapshot_rules(spec)
    example_ids = ordered_example_ids(spec, examples)
    filters = FilterSnapshot(
        min_price=json_number(spec.filters.min_price) if spec.filters.min_price is not None else None,
        min_dollar_vol_20d=(
            json_number(spec.filters.min_dollar_vol_20d)
            if spec.filters.min_dollar_vol_20d is not None
            else None
        ),
        min_adr_pct_20=(
            json_number(spec.filters.min_adr_pct_20) if spec.filters.min_adr_pct_20 is not None else None
        ),
        min_market_cap=json_number(spec.filters.min_market_cap)
        if spec.filters.min_market_cap is not None
        else None,
        max_market_cap=json_number(spec.filters.max_market_cap)
        if spec.filters.max_market_cap is not None
        else None,
    )
    yaml_digest = digest_bytes(bytes(yaml_bytes))
    content = {
        "id": spec.id,
        "name": spec.name,
        "enabled": bool(spec.enabled),
        "timeframe": spec.timeframe,
        "lookback_bars": int(spec.lookback_bars),
        "description": spec.description,
        "llm_notes": spec.llm_notes,
        "rules": [{"rule_id": r.rule_id, "kind": r.kind, "index": r.index, "text": r.text} for r in rules],
        "filters": {
            "min_price": filters.min_price,
            "min_dollar_vol_20d": filters.min_dollar_vol_20d,
            "min_adr_pct_20": filters.min_adr_pct_20,
            "min_market_cap": filters.min_market_cap,
            "max_market_cap": filters.max_market_cap,
        },
        "chart": {"volume": bool(spec.chart.volume), "moving_averages": list(spec.chart.moving_averages)},
        "example_ids": list(example_ids),
    }
    return SetupSnapshot(
        setup_id=spec.id,
        name=spec.name,
        enabled=bool(spec.enabled),
        timeframe=spec.timeframe,
        lookback_bars=int(spec.lookback_bars),
        description=spec.description,
        llm_notes=spec.llm_notes,
        rules=rules,
        filters=filters,
        chart_volume=bool(spec.chart.volume),
        chart_moving_averages=tuple(int(x) for x in spec.chart.moving_averages),
        yaml_bytes=bytes(yaml_bytes),
        yaml_digest=yaml_digest,
        content_digest=digest(content),
        example_ids=example_ids,
    )


def snapshot_example(
    spec: SetupSpec,
    example: VisionExample,
    *,
    image_bytes: bytes | None = None,
    window: BarWindow | None = None,
) -> ExampleInput:
    if example.type == "image":
        if not image_bytes:
            raise VisionError(f"image example {example.id!r} on {spec.id} is missing bytes")
        raw = digest_bytes(image_bytes)
        win = None
        img = bytes(image_bytes)
    else:
        if window is None:
            raise VisionError(f"market_window example {example.id!r} on {spec.id} is missing bars")
        raw = window.source_digest
        win = window
        img = None
    return ExampleInput(
        setup_id=spec.id,
        example_id=example.id,
        polarity=example.polarity,  # type: ignore[arg-type]
        quality=example.quality,
        note=example.note,
        type=example.type,  # type: ignore[arg-type]
        ticker=example.ticker,
        asof_date=example.date,
        timeframe=example.timeframe,
        image_bytes=img,
        window=win,
        raw_digest=raw,
    )


def resolve_chart_profile(setups: Sequence[SetupSpec]) -> ChartProfile:
    """One compatible profile per run. Saved settings win; defaults are not overrides."""
    enabled = [s for s in setups if s.enabled]
    if not enabled:
        raise VisionError("No enabled setups to resolve a chart profile")
    fields = (
        ("timeframe", lambda s: s.timeframe),
        ("lookback_bars", lambda s: int(s.lookback_bars)),
        ("chart.volume", lambda s: bool(s.chart.volume)),
        ("chart.moving_averages", lambda s: tuple(int(x) for x in s.chart.moving_averages)),
    )
    conflicts: list[ProfileFieldConflict] = []
    resolved: dict[str, Any] = {}
    for name, getter in fields:
        values = [(s.id, getter(s)) for s in enabled]
        unique = { _freeze(v) for _, v in values }
        if len(unique) > 1:
            conflicts.append(ProfileFieldConflict(field=name, setup_values=tuple(values)))
        else:
            resolved[name] = values[0][1]
    if conflicts:
        raise ChartProfileConflictError(tuple(conflicts))
    return ChartProfile(
        timeframe=str(resolved["timeframe"]),
        lookback_bars=int(resolved["lookback_bars"]),
        volume=bool(resolved["chart.volume"]),
        moving_averages=tuple(int(x) for x in resolved["chart.moving_averages"]),
    )


def profile_from_style(
    style: ChartStyle,
    *,
    timeframe: str = "1d",
    lookback_bars: int,
) -> ChartProfile:
    return ChartProfile(
        timeframe=timeframe,
        lookback_bars=int(lookback_bars),
        volume=bool(style.volume),
        moving_averages=tuple(int(x) for x in style.moving_averages),
    )


def moving_average_availability(bar_count: int, periods: Sequence[int]) -> tuple[MaAvailability, ...]:
    """Displayed bars only; min_periods=period. Periods longer than the window stay configured."""
    n = int(bar_count)
    out: list[MaAvailability] = []
    for raw in periods:
        period = int(raw)
        available = period > 0 and n >= period
        out.append(
            MaAvailability(
                period=period,
                available=available,
                first_valid_index=(period - 1) if available else None,
                note=(
                    f"SMA {period} uses the {n} displayed bars with min_periods={period}"
                    if available
                    else f"SMA {period} is configured but unavailable on {n} displayed bars"
                ),
            )
        )
    return tuple(out)


def assert_market_cap_unused_for_scan(
    setups: Sequence[SetupSpec],
    *,
    market_cap_available: bool = False,
) -> None:
    for spec in setups:
        if spec.filters.requires_market_cap() and not market_cap_available:
            raise MarketCapUnavailableError(
                "Setup requires market_cap but market-cap data source is unavailable."
            )


def feature_value_from_mapping(row: Mapping[str, Any]) -> FeatureValue:
    return FeatureValue(
        close=json_number(row.get("close")),
        dollar_vol_avg_20=json_number(row.get("dollar_vol_avg_20")),
        adr_pct_20=json_number(row.get("adr_pct_20")),
    )


def bars_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Bar, ...]:
    bars: list[Bar] = []
    for row in rows:
        raw_date = row["date"]
        if isinstance(raw_date, datetime):
            d = raw_date.date()
        elif isinstance(raw_date, date):
            d = raw_date
        else:
            d = date.fromisoformat(str(raw_date)[:10])
        bars.append(
            Bar(
                date=d,
                open=json_number(row.get("open")),
                high=json_number(row.get("high")),
                low=json_number(row.get("low")),
                close=json_number(row.get("close")),
                volume=json_number(row.get("volume")),
            )
        )
    return tuple(bars)


def make_bar_window(
    ticker: str,
    bars: Sequence[Bar],
    *,
    timeframe: str = "1d",
    provenance: str = "unknown",
) -> BarWindow:
    payload = [
        {
            "date": b.date.isoformat(),
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    return BarWindow(
        ticker=str(ticker).strip().upper(),
        timeframe=timeframe,
        bars=tuple(bars),
        provenance=provenance,
        source_digest=digest(payload),
    )


def candidate_identity_payload(
    *,
    ticker: str,
    asof_date: date,
    profile: ChartProfile,
) -> dict[str, Any]:
    return {
        "ticker": str(ticker).strip().upper(),
        "timeframe": profile.timeframe,
        "asof_date": asof_date.isoformat(),
        "lookback_bars": int(profile.lookback_bars),
        "volume": bool(profile.volume),
        "moving_averages": list(profile.moving_averages),
    }


def make_candidate_id(*, ticker: str, asof_date: date, profile: ChartProfile) -> str:
    ident = digest(candidate_identity_payload(ticker=ticker, asof_date=asof_date, profile=profile))
    short = ident.split(":", 1)[1][:12]
    return f"{str(ticker).strip().upper()}:{asof_date.isoformat()}:{short}"


def make_candidate_input(
    *,
    ticker: str,
    window: BarWindow,
    features: FeatureValue,
    eligible_setup_ids: Sequence[str],
    profile: ChartProfile,
) -> CandidateInput:
    asof = window.asof_date
    if asof is None:
        raise VisionError(f"candidate {ticker} has an empty bar window")
    cid = make_candidate_id(ticker=ticker, asof_date=asof, profile=profile)
    return CandidateInput(
        candidate_id=cid,
        ticker=str(ticker).strip().upper(),
        window=window,
        features=features,
        eligible_setup_ids=tuple(eligible_setup_ids),
        source_digest=window.source_digest,
        asof_date=asof,
        profile=profile,
    )


def freeze_prepared_scan(
    *,
    profile: ChartProfile,
    setups: Sequence[SetupSnapshot],
    examples: Sequence[ExampleInput],
    candidates: Sequence[CandidateInput],
    config: ScanConfig,
    diagnostics: ScanDiagnostics | None = None,
    skips: Sequence[InputSkip] = (),
    global_filters_yaml_bytes: bytes = b"",
    prepared_at: datetime | None = None,
    compiler=None,
) -> PreparedScan:
    setups_t = tuple(setups)
    examples_t = tuple(examples)
    candidates_t = tuple(candidates)
    skips_t = tuple(skips)
    at = prepared_at or datetime.now(timezone.utc)
    if compiler is None:
        from .prompts import SnapshotRequestCompiler

        compiler = SnapshotRequestCompiler().snapshot()
    semantic = digest(
        {
            "profile": profile,
            "setups": [{"setup_id": s.setup_id, "content_digest": s.content_digest} for s in setups_t],
            "examples": [
                {
                    "scoped_id": e.scoped_id,
                    "polarity": e.polarity,
                    "quality": e.quality,
                    "note": e.note,
                    "type": e.type,
                    "ticker": e.ticker,
                    "asof_date": e.asof_date,
                    "raw_digest": e.raw_digest,
                }
                for e in examples_t
            ],
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "eligible_setup_ids": list(c.eligible_setup_ids),
                    "asof_date": c.asof_date,
                    "features": c.features,
                    "source_digest": c.source_digest,
                }
                for c in candidates_t
            ],
            "skips": skips_t,
            "config": {
                "model": config.model,
                "timeout_seconds": config.timeout_seconds,
                "max_output_tokens": config.max_output_tokens,
                "image_detail": config.image_detail,
                "mode": config.mode,
            },
            "compiler": compiler.fingerprint if compiler is not None else None,
            "diagnostics": diagnostics or ScanDiagnostics(),
        }
    )
    raw = digest(
        {
            "setup_yaml": [s.yaml_digest for s in setups_t],
            "example_raw": [e.raw_digest for e in examples_t],
            "candidate_source": [c.source_digest for c in candidates_t],
            "global_filters": digest_bytes(bytes(global_filters_yaml_bytes)),
        }
    )
    return PreparedScan(
        profile=profile,
        setups=setups_t,
        examples=examples_t,
        candidates=candidates_t,
        skips=skips_t,
        config=config,
        diagnostics=diagnostics or ScanDiagnostics(),
        prepared_at=at,
        semantic_digest=semantic,
        raw_digest=raw,
        global_filters_yaml_bytes=bytes(global_filters_yaml_bytes),
        global_filters_raw_digest=digest_bytes(bytes(global_filters_yaml_bytes)),
        compiler=compiler,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value
