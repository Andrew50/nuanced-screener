from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from screener_loader.setups.prompt import compile_prompt
from screener_loader.setups.spec import MarketCapUnavailableError
from screener_loader.vision.serialization import (
    SerializationError,
    canonical_dumps,
    digest,
    json_number,
)
from screener_loader.vision.snapshots import (
    assert_market_cap_unused_for_scan,
    freeze_prepared_scan,
    make_bar_window,
    make_candidate_id,
    make_candidate_input,
    moving_average_availability,
    ordered_example_ids,
    resolve_chart_profile,
    rule_id,
    scoped_example_id,
    snapshot_setup,
)
from screener_loader.vision.types import (
    LOOKBACK_BARS_UI_MAX,
    ChartProfileConflictError,
    InputSkip,
    PageRequest,
    ResultQuery,
    ScanConfig,
    ScanDiagnostics,
    StaleSnapshotError,
    load_response_schema,
)
from vision_support import (
    FakeClassifier,
    FakeCompiler,
    FakeRenderer,
    FakeResultReader,
    InMemoryRunStore,
    SYNTHETIC_PNG,
    assessment,
    default_profile,
    feature_value_from_mapping,
    make_example,
    make_spec,
    sample_prepared_scan,
    snapshot_pair,
    window_for,
    write_synthetic_png,
    yaml_bytes_for,
)


def test_vision_chart_profile_equality_and_conflicts() -> None:
    a = make_spec("flag", lookback_bars=100, volume=True, moving_averages=(10, 20, 50))
    b = make_spec("ep", lookback_bars=100, volume=True, moving_averages=(10, 20, 50))
    profile = resolve_chart_profile([a, b])
    assert profile.lookback_bars == 100
    assert profile.moving_averages == (10, 20, 50)

    disabled = make_spec("mr", lookback_bars=80, enabled=False)
    assert resolve_chart_profile([a, disabled]).lookback_bars == 100

    other = make_spec("ep", lookback_bars=80, volume=False, moving_averages=(20, 50))
    with pytest.raises(ChartProfileConflictError) as exc:
        resolve_chart_profile([a, other])
    fields = {c.field for c in exc.value.conflicts}
    assert "lookback_bars" in fields
    assert "chart.volume" in fields
    assert "chart.moving_averages" in fields
    text = str(exc.value)
    assert "flag=100" in text and "ep=80" in text


def test_vision_nondefault_lookback_is_not_overridden() -> None:
    spec = make_spec("flag", lookback_bars=80)
    profile = resolve_chart_profile([spec])
    assert profile.lookback_bars == 80
    assert profile.lookback_bars != 100
    assert 2 <= profile.lookback_bars <= LOOKBACK_BARS_UI_MAX


def test_vision_sma_longer_than_display_is_retained_and_unavailable() -> None:
    avail = moving_average_availability(40, (10, 20, 50, 200))
    by_period = {a.period: a for a in avail}
    assert by_period[10].available is True
    assert by_period[20].available is True
    assert by_period[50].available is False
    assert by_period[200].available is False
    assert by_period[10].first_valid_index == 9
    assert by_period[50].first_valid_index is None
    assert [a.period for a in avail] == [10, 20, 50, 200]


def test_vision_all_examples_quality_ordering_matches_compile_prompt() -> None:
    spec = make_spec("flag")
    examples = [
        make_example("z_edge", quality="edge_case", polarity="positive"),
        make_example("a_near", quality="near_miss", polarity="negative", ticker="XYZ"),
        make_example("m_canon", quality="canonical", polarity="positive"),
        make_example("b_decent", quality="decent", polarity="positive", ticker="AAPL"),
        make_example("n_unspec", quality=None, polarity="negative", ticker="MSFT"),
        make_example("upload", kind="image", polarity="positive", quality="canonical"),
    ]
    ordered = ordered_example_ids(spec, examples)
    assert ordered == compile_prompt(spec, examples).example_ids
    assert ordered == ("m_canon", "upload", "b_decent", "z_edge", "a_near", "n_unspec")
    snap = snapshot_setup(spec, yaml_bytes=yaml_bytes_for(spec), examples=examples)
    assert snap.example_ids == ordered


def test_vision_positional_rule_ids_and_reordering() -> None:
    spec = make_spec(
        "flag",
        required=("tight flag", "dry-up"),
        preferred=("gap",),
        disqualifiers=("extended", "choppy"),
    )
    snap = snapshot_setup(spec, yaml_bytes=yaml_bytes_for(spec), examples=[])
    assert [r.rule_id for r in snap.rules] == [
        "flag:required:0",
        "flag:required:1",
        "flag:preferred:0",
        "flag:disqualifier:0",
        "flag:disqualifier:1",
    ]
    reordered = make_spec(
        "flag",
        required=("dry-up", "tight flag"),
        preferred=("gap",),
        disqualifiers=("extended", "choppy"),
    )
    snap2 = snapshot_setup(reordered, yaml_bytes=yaml_bytes_for(reordered), examples=[])
    assert snap2.rules[0].text == "dry-up"
    assert snap2.rules[0].rule_id == "flag:required:0"
    assert snap.rules[0].text == "tight flag"
    assert snap.content_digest != snap2.content_digest
    assert rule_id("flag", "required", 0) == "flag:required:0"


def test_vision_repeated_example_ids_are_scoped_per_setup() -> None:
    flag = make_spec("flag")
    ep = make_spec("ep", required=("gap",))
    shared = make_example("nvda_2026_06_18", ticker="NVDA")
    flag_snap, flag_ex = snapshot_pair(flag, [shared])
    ep_snap, ep_ex = snapshot_pair(ep, [shared])
    assert flag_ex[0].example_id == ep_ex[0].example_id == "nvda_2026_06_18"
    assert flag_ex[0].scoped_id != ep_ex[0].scoped_id
    assert scoped_example_id("flag", "nvda_2026_06_18") == "flag/nvda_2026_06_18"
    assert flag_snap.setup_id == "flag" and ep_snap.setup_id == "ep"


def test_vision_null_adr_and_market_cap_rejection() -> None:
    feats = feature_value_from_mapping(
        {"close": 10.0, "dollar_vol_avg_20": 8_000_000.0, "adr_pct_20": None}
    )
    assert feats.adr_pct_20 is None
    assert feats.units()["adr_pct_20"] == "fraction"
    import math

    nan_feats = feature_value_from_mapping(
        {"close": 10.0, "dollar_vol_avg_20": float("nan"), "adr_pct_20": math.nan}
    )
    assert nan_feats.dollar_vol_avg_20 is None
    assert nan_feats.adr_pct_20 is None

    capped = make_spec("flag", min_market_cap=100_000_000)
    with pytest.raises(MarketCapUnavailableError):
        assert_market_cap_unused_for_scan([capped], market_cap_available=False)


def test_vision_unknown_diagnostics_are_not_invented() -> None:
    diag = ScanDiagnostics(
        universe_count=1000,
        input_count=800,
        eligible_union_count=12,
        per_setup_eligible_counts=(("flag", 10), ("ep", 5)),
        not_eligible_count=None,
        skipped_count=None,
        unavailable=("filter_rejection_reasons", "not_eligible_count"),
    )
    assert diag.not_eligible_count is None
    assert "not_eligible_count" in diag.unavailable
    # Membership missing is not derived as universe - eligible.
    assert diag.universe_count - diag.eligible_union_count != diag.not_eligible_count


def test_vision_stale_and_short_input_records() -> None:
    skip_short = InputSkip(ticker="ABC", kind="short_window", message="12 bars < lookback 80", bar_count=12)
    skip_stale = InputSkip(ticker="DEF", kind="stale", message="last session 2026-06-10 behind expected 2026-06-18")
    prepared = sample_prepared_scan(skips=(skip_short, skip_stale), diagnostics=ScanDiagnostics(unavailable=("freshness_isolated",)))
    assert prepared.skips[0].kind == "short_window"
    assert prepared.skips[1].kind == "stale"
    err = StaleSnapshotError("global snapshot date is in the future")
    assert "future" in str(err)


def test_vision_candidate_identity_ignores_source_bytes() -> None:
    from screener_loader.vision.snapshots import bars_from_rows

    profile = default_profile(lookback_bars=80)
    w1 = window_for("AAPL", n=20, asof=date(2026, 6, 18))
    shifted = bars_from_rows(
        [
            {
                "date": b.date,
                "open": (b.open or 0) + 9,
                "high": (b.high or 0) + 9,
                "low": (b.low or 0) + 9,
                "close": (b.close or 0) + 9,
                "volume": b.volume,
            }
            for b in w1.bars
        ]
    )
    w2 = make_bar_window("AAPL", shifted)
    id1 = make_candidate_id(ticker="AAPL", asof_date=date(2026, 6, 18), profile=profile)
    id2 = make_candidate_id(ticker="AAPL", asof_date=date(2026, 6, 18), profile=profile)
    assert id1 == id2
    assert w1.source_digest != w2.source_digest
    feats1 = feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04})
    feats2 = feature_value_from_mapping({"close": 19.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04})
    c1 = make_candidate_input(
        ticker="AAPL", window=w1, features=feats1, eligible_setup_ids=("flag",), profile=profile
    )
    c2 = make_candidate_input(
        ticker="AAPL", window=w2, features=feats2, eligible_setup_ids=("flag",), profile=profile
    )
    assert c1.candidate_id == c2.candidate_id
    assert c1.source_digest != c2.source_digest


def test_vision_serialization_sorts_keys_rejects_inf_maps_nan() -> None:
    payload = {"z": 1, "a": [2, 3], "m": {"b": 1, "a": 2}}
    dumped = canonical_dumps(payload).decode("utf-8")
    assert dumped.index('"a"') < dumped.index('"m"') < dumped.index('"z"')
    inner = dumped[dumped.index('"m":') :]
    assert inner.index('"a"') < inner.index('"b"')
    assert json_number(float("nan")) is None
    with pytest.raises(SerializationError):
        json_number(float("inf"))
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_vision_response_schema_and_synthetic_png(tmp_path: Path) -> None:
    schema = load_response_schema()
    assert schema["required"] == ["results"]
    props = schema["properties"]["results"]["items"]["properties"]["assessments"]["items"]["properties"]
    assert set(props["verdict"]["enum"]) == {"match", "no_match", "uncertain"}
    png = write_synthetic_png(tmp_path / "synthetic.png")
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert SYNTHETIC_PNG[:8] == b"\x89PNG\r\n\x1a\n"
    write_synthetic_png()
    assert (Path(__file__).resolve().parent / "fixtures" / "vision" / "synthetic.png").exists()


def test_vision_multiple_no_match_uncertain_error_partial_models() -> None:
    prepared = sample_prepared_scan()
    cid = prepared.candidates[0].candidate_id
    match_a = assessment("flag", "match", strength=2)
    none = assessment("flag", "no_match", violated=("flag:required:0",))
    unsure = assessment("flag", "uncertain", missing=("right edge not visible",))
    assert match_a.verdict == "match" and match_a.match_strength == 2
    assert none.match_strength is None
    assert unsure.verdict == "uncertain"
    store = InMemoryRunStore()
    run = store.create_run(prepared)
    from screener_loader.vision.types import CandidateResult

    store.mark_candidates(
        run.run_id,
        [
            CandidateResult(
                candidate_id=cid,
                status="completed",
                ticker="AAPL",
                asof_date=prepared.candidates[0].asof_date,
                features=prepared.candidates[0].features,
                eligible_setup_ids=("flag", "ep"),
                assessments=(match_a, unsure),
                artifact_id=None,
                error=None,
                attempt_id="a1",
                source_digest=prepared.candidates[0].source_digest,
            )
        ],
    )
    reader = FakeResultReader(store)
    matches = reader.query(ResultQuery(run_id=run.run_id, verdicts=("match",)))
    uncertain = reader.query(ResultQuery(run_id=run.run_id, verdicts=("uncertain",)))
    assert matches.total_candidates == 1
    assert uncertain.total_candidates == 1
    assert matches.items[0].candidate_id == uncertain.items[0].candidate_id
    rec = reader.add_review(run.run_id, cid, judgment="unsure", note="need another session")
    rec2 = reader.add_review(run.run_id, cid, judgment="agree")
    assert rec2.supersedes_review_id == rec.review_id
    assert reader.current_review(run.run_id, cid).judgment == "agree"
    assert reader.list_reviews(run.run_id, cid)[0].judgment == "unsure"


def test_vision_protocol_fakes_do_not_import_unfinished_siblings() -> None:
    import screener_loader.vision as vision

    assert not hasattr(vision, "OpenAIClassifier") or "client" not in vision.__dict__
    renderer = FakeRenderer()
    compiler = FakeCompiler()
    classifier = FakeClassifier()
    store = InMemoryRunStore()
    prepared = sample_prepared_scan()
    art = renderer.render(prepared.candidates[0].window, prepared.profile, title="AAPL")
    assert art.image.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    compiled = compiler.compile(
        setups=prepared.setups,
        example_artifacts=(),
        examples=prepared.examples,
        candidate_artifacts=(art,),
        candidates=prepared.candidates,
        config=prepared.config,
        batch_id="b1",
    )
    attempt = classifier.classify(compiled)
    assert attempt.provider == "fake"
    run = store.create_run(prepared)
    with store.lock_run(run.run_id):
        store.save_artifact(run.run_id, art)
        store.journal_attempt(run.run_id, attempt)
    assert store.get_artifact_bytes(run.run_id, art.artifact_id)[:8] == b"\x89PNG\r\n\x1a\n"
    page = FakeResultReader(store).query(ResultQuery(run_id=run.run_id, page=PageRequest(page=1, page_size=10)))
    assert page.has_prev is False
    cfg = ScanConfig(model="gpt-4.1-2025-04-14")
    assert cfg.batch_size == 10 and cfg.max_concurrency == 16


def test_vision_semantic_vs_raw_digest_resume_role() -> None:
    a = sample_prepared_scan()
    b = sample_prepared_scan()
    assert a.semantic_digest == b.semantic_digest
    spec = make_spec("flag", required=("other rule",))
    snap, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    changed = freeze_prepared_scan(
        profile=a.profile,
        setups=(snap,),
        examples=examples,
        candidates=a.candidates,
        config=a.config,
        diagnostics=a.diagnostics,
        global_filters_yaml_bytes=a.global_filters_yaml_bytes,
    )
    assert changed.semantic_digest != a.semantic_digest
    # Whitespace-only YAML changes raw digest of that setup, not content_digest.
    snap2 = snapshot_setup(make_spec("flag"), yaml_bytes=yaml_bytes_for(make_spec("flag")) + b"\n", examples=[])
    snap3 = snapshot_setup(make_spec("flag"), yaml_bytes=yaml_bytes_for(make_spec("flag")), examples=[])
    assert snap2.content_digest == snap3.content_digest
    assert snap2.yaml_digest != snap3.yaml_digest
