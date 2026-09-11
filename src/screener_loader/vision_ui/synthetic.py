"""Synthetic demo/test scan data. Labeled fake results; never used as a production fallback."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4
import struct
import zlib

from screener_loader.setups.spec import ChartStyle, SetupCriteria, SetupFilters, SetupSpec, VisionExample
from screener_loader.vision.query import FilesystemResultReader
from screener_loader.vision.reviews import FilesystemReviewStore
from screener_loader.vision.serialization import digest_bytes
from screener_loader.vision.snapshots import (
    bars_from_rows,
    feature_value_from_mapping,
    freeze_prepared_scan,
    make_bar_window,
    make_candidate_input,
    snapshot_example,
    snapshot_setup,
)
from screener_loader.vision.store import FilesystemRunStore, open_vision_scans
from screener_loader.vision.types import (
    Assessment,
    AttemptError,
    CandidateResult,
    ChartArtifact,
    ChartProfile,
    ClassificationAttempt,
    ExampleInput,
    FeatureValue,
    InputSkip,
    PreparedScan,
    RenderedImage,
    RunSummary,
    ScanConfig,
    ScanDiagnostics,
    SetupSnapshot,
    TokenUsage,
)


def synthetic_png_bytes(width: int = 8, height: int = 6, color: tuple[int, int, int] = (20, 80, 160)) -> bytes:
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


SYNTHETIC_PNG = synthetic_png_bytes()


def _spec(setup_id: str, name: str, required: tuple[str, ...]) -> SetupSpec:
    return SetupSpec(
        id=setup_id,
        name=name,
        enabled=True,
        lookback_bars=100,
        description=f"Synthetic {name} definition frozen into the demo run.",
        criteria=SetupCriteria(required=required, preferred=("volume dry-up",), disqualifiers=("too extended",)),
        filters=SetupFilters(min_adr_pct_20=0.03),
        chart=ChartStyle(volume=True, moving_averages=(10, 20, 50)),
        llm_notes="Demo snapshot. Not a live catalog edit.",
    )


def _example(example_id: str, *, polarity: str = "positive", quality: str | None = "canonical", ticker: str = "NVDA") -> VisionExample:
    return VisionExample(
        id=example_id,
        polarity=polarity,  # type: ignore[arg-type]
        type="market_window",
        quality=quality,  # type: ignore[arg-type]
        note="synthetic example",
        ticker=ticker,
        date=date(2026, 6, 18),
    )


def _upload(example_id: str) -> VisionExample:
    return VisionExample(
        id=example_id,
        polarity="negative",  # type: ignore[arg-type]
        type="image",
        quality="edge_case",  # type: ignore[arg-type]
        note="uploaded synthetic png",
        path="examples/synthetic.png",
    )


def _yaml(spec: SetupSpec) -> bytes:
    return (
        f"id: {spec.id}\nname: {spec.name}\nlookback_bars: {spec.lookback_bars}\n"
        f"chart.volume: true\nchart.moving_averages: [10, 20, 50]\n"
    ).encode("utf-8")


def _rows(ticker: str, *, asof: date = date(2026, 6, 18), n: int = 30, close0: float = 100.0) -> list[dict]:
    start = asof - timedelta(days=n - 1)
    out = []
    for i in range(n):
        d = start + timedelta(days=i)
        px = close0 + i * 0.4
        out.append(
            {
                "ticker": ticker,
                "date": d,
                "open": px - 0.2,
                "high": px + 0.8,
                "low": px - 0.7,
                "close": px,
                "volume": 1_000_000.0 + i,
            }
        )
    return out


def _window(ticker: str, *, n: int = 30, asof: date = date(2026, 6, 18), close0: float = 100.0):
    return make_bar_window(ticker, bars_from_rows(_rows(ticker, asof=asof, n=n, close0=close0)), provenance="unknown")


def _profile() -> ChartProfile:
    return ChartProfile(timeframe="1d", lookback_bars=100, volume=True, moving_averages=(10, 20, 50))


def _features(*, close: float | None = 12.5, dv: float | None = 8_000_000.0, adr: float | None = 0.05) -> FeatureValue:
    return feature_value_from_mapping({"close": close, "dollar_vol_avg_20": dv, "adr_pct_20": adr})


def _assessment(setup_id: str, verdict: str, *, strength: int | None = None, reason: str = "Synthetic coiled structure.", violated: tuple[str, ...] = (), missing: tuple[str, ...] = ()) -> Assessment:
    return Assessment(
        setup_id=setup_id,
        verdict=verdict,  # type: ignore[arg-type]
        match_strength=strength,
        reason=reason,
        violated_required_rule_ids=violated,
        missing_evidence=missing,
    )


def build_two_setup_scan(
    *,
    tickers: Sequence[str],
    mode: str = "demo",
    asof: date = date(2026, 6, 18),
    diagnostics: ScanDiagnostics | None = None,
    skips: Sequence[InputSkip] = (),
    extra_candidates: Sequence = (),
) -> PreparedScan:
    flag = _spec("flag", "Bull Flag", ("tight flag",))
    ep = _spec("ep", "Episodic Pivot", ("gap and hold",))
    flag_examples = [
        _example("ex_canonical", quality="canonical"),
        _example("ex_near", quality="near_miss", polarity="negative", ticker="XYZ"),
        _upload("upload_1"),
    ]
    ep_examples = [_example("ep_canon", quality="canonical", ticker="AMD")]
    flag_snap = snapshot_setup(flag, yaml_bytes=_yaml(flag), examples=flag_examples)
    ep_snap = snapshot_setup(ep, yaml_bytes=_yaml(ep), examples=ep_examples)
    examples: list[ExampleInput] = []
    for spec, snap_examples in ((flag, flag_examples), (ep, ep_examples)):
        for ex in snap_examples:
            if ex.type == "image":
                examples.append(snapshot_example(spec, ex, image_bytes=SYNTHETIC_PNG))
            else:
                examples.append(snapshot_example(spec, ex, window=_window(ex.ticker or "NVDA", asof=asof)))
    profile = _profile()
    candidates = []
    for i, ticker in enumerate(tickers):
        eligible = ("flag", "ep") if i % 5 != 4 else ("flag",)
        candidates.append(
            make_candidate_input(
                ticker=ticker,
                window=_window(ticker, asof=asof, close0=10.0 + i),
                features=_features(close=10.0 + i, dv=5_000_000.0 + i * 10_000, adr=0.04 + (i % 4) * 0.01),
                eligible_setup_ids=eligible,
                profile=profile,
            )
        )
    candidates.extend(extra_candidates)
    return freeze_prepared_scan(
        profile=profile,
        setups=(flag_snap, ep_snap),
        examples=tuple(examples),
        candidates=tuple(candidates),
        config=ScanConfig(model="synthetic-demo-model", mode=mode),  # type: ignore[arg-type]
        diagnostics=diagnostics
        or ScanDiagnostics(
            universe_count=40,
            input_count=30,
            eligible_union_count=len(candidates),
            per_setup_eligible_counts=(("flag", len(candidates)), ("ep", max(0, len(candidates) - 1))),
            not_eligible_count=None,
            skipped_count=None,
            unavailable=("filter_rejection_reasons", "not_eligible_count"),
        ),
        skips=tuple(skips),
        global_filters_yaml_bytes=b"min_price: 2.0\nmin_dollar_vol_20d: 5000000\n",
        prepared_at=datetime(2026, 6, 19, 14, 0, tzinfo=timezone.utc),
    )


def _chart(candidate, png: bytes, *, color: tuple[int, int, int] | None = None) -> ChartArtifact:
    data = synthetic_png_bytes(color=color) if color else png
    return ChartArtifact(
        artifact_id=f"candidate:{candidate.ticker}:{candidate.asof_date.isoformat()}",
        kind="candidate",
        image=RenderedImage(png_bytes=data, width=8, height=6, sha256=digest_bytes(data)),
        title=f"{candidate.ticker}  {candidate.asof_date.isoformat()}",
        profile=candidate.profile,
        ma_availability=(),
        final_session_date=candidate.asof_date,
        ticker=candidate.ticker,
        candidate_id=candidate.candidate_id,
        source="rendered",
    )


def _attempt(batch_id: str, candidate_ids: Sequence[str], results: Sequence[CandidateResult]) -> ClassificationAttempt:
    now = datetime(2026, 6, 19, 15, 0, tzinfo=timezone.utc)
    return ClassificationAttempt(
        attempt_id=f"att-{uuid4().hex[:12]}",
        batch_id=batch_id,
        batch_fingerprint=f"fp-{batch_id}",
        candidate_ids=tuple(candidate_ids),
        setup_ids=("flag", "ep"),
        started_at=now,
        ended_at=now,
        provider="fake",
        model="synthetic-demo-model",
        response_id=f"fake-{batch_id}",
        usage=TokenUsage(input_tokens=20, output_tokens=40, total_tokens=60),
        retry_after_seconds=None,
        latency_ms=12,
        error=None,
        sanitized_output={"synthetic": True},
        results=None,
        accepted=True,
    )


def _scripted_assessments(index: int, eligible: Sequence[str]) -> tuple[Assessment, ...]:
    out: list[Assessment] = []
    kind = index % 7
    for sid in eligible:
        if index == 0 and sid == "flag":
            out.append(_assessment(sid, "match", strength=3, reason="Tight flag under resistance with dry volume."))
        elif index == 0 and sid == "ep":
            out.append(_assessment(sid, "match", strength=2, reason="Held the opening range after the impulse."))
        elif kind == 1:
            out.append(
                _assessment(
                    sid,
                    "no_match",
                    reason="Base is sloppy and already extended.",
                    violated=(f"{sid}:required:0",),
                )
            )
        elif kind == 2:
            out.append(
                _assessment(sid, "uncertain", reason="Right edge is clipped.", missing=("right edge not visible",))
            )
        elif kind == 5:
            out.append(_assessment(sid, "no_match", reason="No orderly contraction."))
        elif kind == 6:
            out.append(_assessment(sid, "uncertain", reason="Volume context is ambiguous.", missing=("relative volume",)))
        elif kind == 3:
            out.append(_assessment(sid, "match", strength=1, reason="Marginal coil; required rules only barely present."))
        else:
            out.append(_assessment(sid, "match", strength=2, reason="Solid contraction into moving averages."))
    return tuple(out)


def _complete_candidate(store: FilesystemRunStore, run_id: str, cand, assessments: tuple[Assessment, ...], png: bytes) -> None:
    art = store.save_artifact(run_id, _chart(cand, png))
    row = CandidateResult(
        candidate_id=cand.candidate_id,
        status="completed",
        ticker=cand.ticker,
        asof_date=cand.asof_date,
        features=cand.features,
        eligible_setup_ids=cand.eligible_setup_ids,
        assessments=assessments,
        artifact_id=art.artifact_id,
        error=None,
        attempt_id="att-demo",
        source_digest=cand.source_digest,
    )
    attempt = _attempt(f"batch-{cand.ticker}", (cand.candidate_id,), (row,))
    store.commit_batch(run_id, batch_id=f"batch-{cand.ticker.lower()}", results=(row,), attempt=attempt)


def seed_synthetic_runs(scan_root: Path) -> tuple[FilesystemRunStore, FilesystemResultReader, FilesystemReviewStore]:
    """Write clearly synthetic runs into ``scan_root``. Idempotent if the root already has runs."""

    store, reader, reviews = open_vision_scans(scan_root)
    if store.list_runs():
        return store, reader, reviews

    partial_scan = build_two_setup_scan(tickers=("AAA", "BBB", "CCC", "DDD"), mode="demo")
    partial = store.create_run(partial_scan)
    _complete_candidate(
        store,
        partial.run_id,
        partial_scan.candidates[0],
        (_assessment("flag", "match", strength=2), _assessment("ep", "no_match", reason="No range hold.")),
        SYNTHETIC_PNG,
    )
    store.mark_candidates(
        partial.run_id,
        [
            CandidateResult(
                candidate_id=partial_scan.candidates[1].candidate_id,
                status="error",
                ticker=partial_scan.candidates[1].ticker,
                asof_date=partial_scan.candidates[1].asof_date,
                features=partial_scan.candidates[1].features,
                eligible_setup_ids=partial_scan.candidates[1].eligible_setup_ids,
                assessments=(),
                artifact_id=None,
                error=AttemptError(kind="timeout", message="Synthetic timeout; not a no-match.", retryable=True),
                attempt_id="att-timeout",
                source_digest=partial_scan.candidates[1].source_digest,
            )
        ],
    )
    store.save_artifact(partial.run_id, _chart(partial_scan.candidates[2], SYNTHETIC_PNG))
    store.finalize(
        partial.run_id,
        RunSummary(
            candidates_total=4,
            candidates_completed=1,
            candidates_error=1,
            candidates_skipped=0,
            candidates_pending=2,
            setup_matches=1,
            attempts=2,
            usage=None,
            status="partial",
            synthetic=True,
        ),
    )

    empty = build_two_setup_scan(tickers=(), mode="demo")
    empty_run = store.create_run(empty)
    store.finalize(
        empty_run.run_id,
        RunSummary(
            candidates_total=0,
            candidates_completed=0,
            candidates_error=0,
            candidates_skipped=0,
            candidates_pending=0,
            setup_matches=0,
            attempts=0,
            usage=None,
            status="completed",
            synthetic=True,
        ),
    )

    dry = build_two_setup_scan(tickers=("DRY1", "DRY2"), mode="dry_run")
    dry_run = store.create_run(dry)
    for cand in dry.candidates:
        store.save_artifact(dry_run.run_id, _chart(cand, SYNTHETIC_PNG, color=(80, 20, 20)))
    store.finalize(
        dry_run.run_id,
        RunSummary(
            candidates_total=2,
            candidates_completed=0,
            candidates_error=0,
            candidates_skipped=0,
            candidates_pending=2,
            setup_matches=0,
            attempts=0,
            usage=None,
            status="dry_run",
            synthetic=True,
        ),
    )

    failed = build_two_setup_scan(tickers=("ZZZ",), mode="demo")
    failed_run = store.create_run(failed)
    store.mark_candidates(
        failed_run.run_id,
        [
            CandidateResult(
                candidate_id=failed.candidates[0].candidate_id,
                status="error",
                ticker=failed.candidates[0].ticker,
                asof_date=failed.candidates[0].asof_date,
                features=failed.candidates[0].features,
                eligible_setup_ids=failed.candidates[0].eligible_setup_ids,
                assessments=(),
                artifact_id=None,
                error=AttemptError(kind="auth", message="Synthetic auth failure. No fake matches.", retryable=False),
                attempt_id=None,
                source_digest=failed.candidates[0].source_digest,
            )
        ],
    )
    store.finalize(
        failed_run.run_id,
        RunSummary(
            candidates_total=1,
            candidates_completed=0,
            candidates_error=1,
            candidates_skipped=0,
            candidates_pending=0,
            setup_matches=0,
            attempts=1,
            usage=TokenUsage(input_tokens=8, output_tokens=0, total_tokens=8),
            status="failed",
            synthetic=True,
        ),
    )
    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMD",
        "AMZN",
        "META",
        "GOOG",
        "NFLX",
        "AVGO",
        "COST",
        "ADBE",
        "PEP",
        "KO",
        "NKE",
        "DIS",
        "INTC",
        "QCOM",
        "TXN",
        "INTU",
        "AMAT",
        "LRCX",
        "NOW",
        "CRWD",
        "PANW",
        "SHOP",
        "UBER",
        "SQ",
    ]
    skips = (
        InputSkip(ticker="SHORT", kind="short_window", message="12 bars < lookback 100", bar_count=12, asof_date=date(2026, 6, 18)),
        InputSkip(ticker="STALE", kind="stale", message="last session 2026-06-10 behind expected 2026-06-18", asof_date=date(2026, 6, 10)),
    )
    main = build_two_setup_scan(tickers=tickers, skips=skips)
    run = store.create_run(main)
    store.update_status(run.run_id, "running")
    for ex in main.examples:
        if ex.image_bytes:
            store.save_artifact(
                run.run_id,
                ChartArtifact(
                    artifact_id=f"example:{ex.scoped_id}",
                    kind="example",
                    image=RenderedImage(
                        png_bytes=ex.image_bytes, width=8, height=6, sha256=digest_bytes(ex.image_bytes)
                    ),
                    title=ex.scoped_id,
                    profile=None,
                    ma_availability=(),
                    final_session_date=ex.asof_date,
                    setup_id=ex.setup_id,
                    example_id=ex.example_id,
                    source="upload",
                ),
            )
    error_ticker = "CRWD"
    missing_chart_ticker = "PANW"
    for i, cand in enumerate(main.candidates):
        if cand.ticker == error_ticker:
            store.mark_candidates(
                run.run_id,
                [
                    CandidateResult(
                        candidate_id=cand.candidate_id,
                        status="error",
                        ticker=cand.ticker,
                        asof_date=cand.asof_date,
                        features=cand.features,
                        eligible_setup_ids=cand.eligible_setup_ids,
                        assessments=(),
                        artifact_id=None,
                        error=AttemptError(
                            kind="unavailable_input",
                            message="Synthetic render failure: chart bytes were not saved.",
                            retryable=False,
                        ),
                        attempt_id=None,
                        source_digest=cand.source_digest,
                    )
                ],
            )
            continue
        if cand.ticker == missing_chart_ticker:
            store.mark_candidates(
                run.run_id,
                [
                    CandidateResult(
                        candidate_id=cand.candidate_id,
                        status="error",
                        ticker=cand.ticker,
                        asof_date=cand.asof_date,
                        features=cand.features,
                        eligible_setup_ids=cand.eligible_setup_ids,
                        assessments=(),
                        artifact_id="missing-chart",
                        error=AttemptError(kind="unavailable_input", message="Saved chart path is missing.", retryable=False),
                        attempt_id=None,
                        source_digest=cand.source_digest,
                    )
                ],
            )
            continue
        _complete_candidate(store, run.run_id, cand, _scripted_assessments(i, cand.eligible_setup_ids), SYNTHETIC_PNG)
    store.mark_candidates(
        run.run_id,
        [
            CandidateResult(
                candidate_id=f"skip:{s.ticker}:{s.kind}",
                status="skipped",
                ticker=s.ticker,
                asof_date=s.asof_date or date(2026, 6, 18),
                features=_features(close=None, dv=None, adr=None),
                eligible_setup_ids=(),
                assessments=(),
                artifact_id=None,
                error=AttemptError(kind=s.kind, message=s.message, retryable=False),  # type: ignore[arg-type]
                attempt_id=None,
                source_digest="",
            )
            for s in skips
        ],
    )
    aapl = next(c for c in main.candidates if c.ticker == "AAPL")
    first = reviews.add_review(run.run_id, aapl.candidate_id, judgment="unsure", note="Need another session.", setup_id="flag")
    reviews.add_review(
        run.run_id,
        aapl.candidate_id,
        judgment="agree",
        note="Confirmed both flags. Synthetic revision.",
        setup_id="flag",
        expected_current_id=first.review_id,
    )
    reviews.add_review(run.run_id, aapl.candidate_id, judgment="disagree", note="EP is a miss.", setup_id="ep")
    store.finalize(
        run.run_id,
        RunSummary(
            candidates_total=len(main.candidates) + 2,
            candidates_completed=len(main.candidates) - 2,
            candidates_error=2,
            candidates_skipped=2,
            candidates_pending=0,
            setup_matches=sum(
                1
                for r in store.list_candidate_results(run.run_id)
                for a in r.assessments
                if a.verdict == "match"
            ),
            attempts=len(main.candidates),
            usage=TokenUsage(input_tokens=400, output_tokens=800, total_tokens=1200),
            status="partial",
            synthetic=True,
        ),
    )

    return store, reader, reviews


def setups_by_id(prepared: PreparedScan) -> dict[str, SetupSnapshot]:
    return {s.setup_id: s for s in prepared.setups}
