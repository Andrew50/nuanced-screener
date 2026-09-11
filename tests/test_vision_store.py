"""Persistence, query, review, and export tests for the vision results layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from screener_loader.vision.query import FilesystemResultReader
from screener_loader.vision.reviews import FilesystemReviewStore
from screener_loader.vision.serialization import digest_bytes
from screener_loader.vision.snapshots import freeze_prepared_scan
from screener_loader.vision.store import (
    SCHEMA_VERSION,
    FilesystemRunStore,
    is_temp_name,
    open_vision_scans,
    read_json,
)
from screener_loader.vision.types import (
    AttemptError,
    CandidateResult,
    ChartArtifact,
    ClassificationAttempt,
    InputSkip,
    PageRequest,
    RenderedImage,
    ResultQuery,
    RunLockedError,
    RunSummary,
    ScanDiagnostics,
    SortSpec,
    TokenUsage,
    VisionError,
)
from screener_loader.vision_ui.components import reconcile_selected_candidate
from screener_loader.vision_ui.synthetic import seed_synthetic_runs
from vision_support import (
    SYNTHETIC_PNG,
    FakeClassifier,
    FakeCompiler,
    FakeRenderer,
    assessment,
    feature_value_from_mapping,
    make_candidate_input,
    make_spec,
    sample_prepared_scan,
    snapshot_pair,
    window_for,
)


def _png_artifact(cand, png: bytes = SYNTHETIC_PNG) -> ChartArtifact:
    return ChartArtifact(
        artifact_id=f"candidate:{cand.ticker}:{cand.asof_date.isoformat()}",
        kind="candidate",
        image=RenderedImage(png_bytes=png, width=8, height=6, sha256=digest_bytes(png)),
        title=f"{cand.ticker} {cand.asof_date}",
        profile=cand.profile,
        ma_availability=(),
        final_session_date=cand.asof_date,
        ticker=cand.ticker,
        candidate_id=cand.candidate_id,
        source="rendered",
    )


def _attempt(batch_id: str, cids: tuple[str, ...], *, accepted: bool = True, at: datetime | None = None) -> ClassificationAttempt:
    now = at or datetime(2026, 6, 19, 16, 0, tzinfo=timezone.utc)
    return ClassificationAttempt(
        attempt_id=f"att-{batch_id}",
        batch_id=batch_id,
        batch_fingerprint=f"fp-{batch_id}",
        candidate_ids=cids,
        setup_ids=("flag",),
        started_at=now,
        ended_at=now,
        provider="fake",
        model="gpt-4.1-2025-04-14",
        response_id=f"resp-{batch_id}",
        usage=TokenUsage(input_tokens=11, output_tokens=22, total_tokens=33),
        retry_after_seconds=None,
        latency_ms=4,
        error=None,
        sanitized_output={"ok": True},
        results=None,
        accepted=accepted,
    )


def _completed(cand, assessments, artifact_id: str | None) -> CandidateResult:
    return CandidateResult(
        candidate_id=cand.candidate_id,
        status="completed",
        ticker=cand.ticker,
        asof_date=cand.asof_date,
        features=cand.features,
        eligible_setup_ids=cand.eligible_setup_ids,
        assessments=assessments,
        artifact_id=artifact_id,
        error=None,
        attempt_id="att-b1",
        source_digest=cand.source_digest,
    )


def test_json_parquet_image_round_trip(tmp_path: Path) -> None:
    store, reader, _reviews = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan()
    cand = prepared.candidates[0]
    run = store.create_run(prepared)
    ref = store.save_artifact(run.run_id, _png_artifact(cand))
    assert ref.relative_path.startswith("artifacts/candidates/")
    assert ".." not in ref.relative_path
    assert store.get_artifact_bytes(run.run_id, ref.artifact_id) == SYNTHETIC_PNG
    loaded = store.load_frozen_inputs(run.run_id)
    assert loaded.semantic_digest == prepared.semantic_digest
    assert loaded.setups[0].yaml_bytes == prepared.setups[0].yaml_bytes
    assert loaded.global_filters_yaml_bytes == prepared.global_filters_yaml_bytes
    upload = next(ex for ex in loaded.examples if ex.type == "image")
    original = next(ex for ex in prepared.examples if ex.type == "image")
    assert upload.image_bytes == original.image_bytes
    assert loaded.candidates[0].window.bars == prepared.candidates[0].window.bars
    row = _completed(cand, (assessment("flag", "match", strength=2),), ref.artifact_id)
    store.commit_batch(run.run_id, batch_id="b1", results=(row,), attempt=_attempt("b1", (cand.candidate_id,)))
    store.finalize(
        run.run_id,
        RunSummary(
            candidates_total=1,
            candidates_completed=1,
            candidates_error=0,
            candidates_skipped=0,
            candidates_pending=0,
            setup_matches=1,
            attempts=1,
            usage=TokenUsage(11, 22, 33),
            status="completed",
        ),
    )
    derived = store.run_dir(run.run_id) / "derived" / "candidates.parquet"
    table = pd.read_parquet(derived)
    assert table.iloc[0]["ticker"] == "AAPL"
    assert table.iloc[0]["adr_pct_20"] == pytest.approx(0.05)
    page = reader.query(ResultQuery(run_id=run.run_id))
    assert page.total_candidates == 1
    assert page.items[0].matched_setups[0].match_strength == 2
    assert reader.get_artifact_bytes(run.run_id, ref.artifact_id)[:8] == b"\x89PNG\r\n\x1a\n"


def test_atomic_commit_recovery_and_temp_ignored(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    prepared = sample_prepared_scan()
    cand = prepared.candidates[0]
    run = store.create_run(prepared)
    row = _completed(cand, (assessment("flag", "no_match"),), None)
    store.commit_batch(run.run_id, batch_id="b1", results=(row,), attempt=_attempt("b1", (cand.candidate_id,)))
    results_path = store.run_dir(run.run_id) / "results"
    result_file = next(results_path.glob("*.json"))
    result_file.unlink()
    recovered = store.list_candidate_results(run.run_id)
    assert recovered[0].candidate_id == cand.candidate_id
    tmp = store.run_dir(run.run_id) / "results" / ".tmp-junk.json"
    tmp.write_text('{"candidate_id": "FAKE"}', encoding="utf-8")
    assert is_temp_name(tmp.name)
    ids = {r.candidate_id for r in store.list_candidate_results(run.run_id)}
    assert "FAKE" not in ids


def test_identical_commit_is_idempotent_conflict_is_rejected(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    prepared = sample_prepared_scan()
    cand = prepared.candidates[0]
    run = store.create_run(prepared)
    row = _completed(cand, (assessment("flag", "match", strength=1),), None)
    attempt = _attempt("b1", (cand.candidate_id,))
    store.commit_batch(run.run_id, batch_id="b1", results=(row,), attempt=attempt)
    store.commit_batch(run.run_id, batch_id="b1", results=(row,), attempt=attempt)
    other = _completed(cand, (assessment("flag", "match", strength=3),), None)
    with pytest.raises(VisionError, match="different content"):
        store.commit_batch(run.run_id, batch_id="b1", results=(other,), attempt=attempt)
    with pytest.raises(VisionError, match="different content"):
        store.commit_batch(run.run_id, batch_id="b2", results=(other,), attempt=_attempt("b2", (cand.candidate_id,)))


def test_process_lock_and_path_escape(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    prepared = sample_prepared_scan()
    run = store.create_run(prepared)
    with store.lock_run(run.run_id):
        with pytest.raises(RunLockedError):
            with store.lock_run(run.run_id):
                pass
    with store.lock_run(run.run_id):
        pass
    with pytest.raises(VisionError):
        store.run_dir("../escape")


def test_unsupported_schema_does_not_overwrite(tmp_path: Path) -> None:
    store = FilesystemRunStore(tmp_path)
    prepared = sample_prepared_scan()
    run = store.create_run(prepared)
    meta_path = store.run_dir(run.run_id) / "meta.json"
    payload = read_json(meta_path)
    payload["schema_version"] = 99
    meta_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    original = meta_path.read_bytes()
    with pytest.raises(VisionError, match="Unsupported"):
        store.load_run(run.run_id)
    assert meta_path.read_bytes() == original
    assert SCHEMA_VERSION == 1


def test_partial_and_dry_run_reads_include_pending(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan(mode="dry_run")
    run = store.create_run(prepared)
    assert store.load_run(run.run_id).status == "dry_run"
    page = reader.query(ResultQuery(run_id=run.run_id))
    assert page.total_candidates == 1
    assert page.items[0].status == "pending"
    counts = reader.counts(run.run_id)
    assert counts.pending == 1
    assert counts.completed == 0


def test_one_row_per_candidate_and_multi_view(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    flag = make_spec("flag")
    ep = make_spec("ep", required=("gap",))
    flag_snap, flag_ex = snapshot_pair(flag, [])
    ep_snap, ep_ex = snapshot_pair(ep, [])
    profile = sample_prepared_scan().profile
    feats = feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04})
    cand = make_candidate_input(
        ticker="AAPL",
        window=window_for("AAPL"),
        features=feats,
        eligible_setup_ids=("flag", "ep"),
        profile=profile,
    )
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(flag_snap, ep_snap),
        examples=flag_ex + ep_ex,
        candidates=(cand,),
        config=sample_prepared_scan().config,
        diagnostics=ScanDiagnostics(universe_count=10, not_eligible_count=None, unavailable=("not_eligible_count",)),
        global_filters_yaml_bytes=b"min_price: 2\n",
    )
    run = store.create_run(prepared)
    store.commit_batch(
        run.run_id,
        batch_id="b1",
        results=(
            _completed(
                cand,
                (
                    assessment("flag", "match", strength=1),
                    assessment("ep", "uncertain", missing=("right edge",)),
                ),
                None,
            ),
        ),
        attempt=_attempt("b1", (cand.candidate_id,)),
    )
    matches = reader.query(ResultQuery(run_id=run.run_id, verdicts=("match",)))
    uncertain = reader.query(ResultQuery(run_id=run.run_id, verdicts=("uncertain",)))
    assert matches.total_candidates == 1
    assert uncertain.total_candidates == 1
    assert matches.items[0].candidate_id == uncertain.items[0].candidate_id
    assert len(matches.items[0].matched_setups) == 1
    ext = reader.extended_counts(run.run_id)
    assert ext.any_match == 1
    assert ext.setup_matches == 1
    assert ext.uncertain == 1
    assert prepared.diagnostics.not_eligible_count is None


def test_setup_aware_strength_sort_and_filter(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    flag = make_spec("flag")
    ep = make_spec("ep", required=("gap",))
    flag_snap, _ = snapshot_pair(flag, [])
    ep_snap, _ = snapshot_pair(ep, [])
    profile = sample_prepared_scan().profile
    feats = feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": None})
    a = make_candidate_input(
        ticker="AAA", window=window_for("AAA"), features=feats, eligible_setup_ids=("flag", "ep"), profile=profile
    )
    b = make_candidate_input(
        ticker="BBB", window=window_for("BBB"), features=feats, eligible_setup_ids=("flag", "ep"), profile=profile
    )
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(flag_snap, ep_snap),
        examples=(),
        candidates=(a, b),
        config=sample_prepared_scan().config,
        global_filters_yaml_bytes=b"",
    )
    run = store.create_run(prepared)
    store.commit_batch(
        run.run_id,
        batch_id="b1",
        results=(
            _completed(a, (assessment("flag", "match", strength=1), assessment("ep", "match", strength=3)), None),
            _completed(b, (assessment("flag", "match", strength=2), assessment("ep", "match", strength=1)), None),
        ),
        attempt=_attempt("b1", (a.candidate_id, b.candidate_id)),
    )
    q = ResultQuery(run_id=run.run_id, setup_ids=("flag",), sort=SortSpec(key="match_strength", descending=True))
    page = reader.query(q)
    assert [r.ticker for r in page.items] == ["BBB", "AAA"]
    filtered = reader.query(ResultQuery(run_id=run.run_id, setup_ids=("flag",), min_match_strength=2))
    assert [r.ticker for r in filtered.items] == ["BBB"]
    export = reader.export_query(q)
    assert export.candidates[0]["adr_pct_20"] is None
    assert export.candidates[0]["adr_pct_20_unit"] == "fraction"
    assert export.flattening


def test_arrival_sort_appends_later_commits(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    flag = make_spec("flag")
    flag_snap, _ = snapshot_pair(flag, [])
    profile = sample_prepared_scan().profile
    feats = feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": None})
    early = make_candidate_input(
        ticker="EARLY", window=window_for("EARLY"), features=feats, eligible_setup_ids=("flag",), profile=profile
    )
    later = make_candidate_input(
        ticker="LATER", window=window_for("LATER"), features=feats, eligible_setup_ids=("flag",), profile=profile
    )
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(flag_snap,),
        examples=(),
        candidates=(early, later),
        config=sample_prepared_scan().config,
        global_filters_yaml_bytes=b"",
    )
    run = store.create_run(prepared)
    t0 = datetime(2026, 6, 19, 16, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 19, 16, 1, tzinfo=timezone.utc)
    store.commit_batch(
        run.run_id,
        batch_id="first",
        results=(_completed(early, (assessment("flag", "match", strength=1),), None),),
        attempt=_attempt("first", (early.candidate_id,), at=t0),
    )
    store.commit_batch(
        run.run_id,
        batch_id="second",
        results=(_completed(later, (assessment("flag", "match", strength=3),), None),),
        attempt=_attempt("second", (later.candidate_id,), at=t1),
    )
    by_arrival = reader.query(ResultQuery(run_id=run.run_id, sort=SortSpec(key="arrival", descending=False)))
    assert [r.ticker for r in by_arrival.items] == ["EARLY", "LATER"]
    by_strength = reader.query(ResultQuery(run_id=run.run_id, sort=SortSpec(key="match_strength", descending=True)))
    assert [r.ticker for r in by_strength.items] == ["LATER", "EARLY"]


def test_pagination_neighbors_and_selection_rule(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan()
    profile = prepared.profile
    feats = prepared.candidates[0].features
    cands = tuple(
        make_candidate_input(
            ticker=f"T{i:02d}",
            window=window_for(f"T{i:02d}"),
            features=feats,
            eligible_setup_ids=("flag",),
            profile=profile,
        )
        for i in range(12)
    )
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=prepared.setups,
        examples=prepared.examples,
        candidates=cands,
        config=prepared.config,
        global_filters_yaml_bytes=prepared.global_filters_yaml_bytes,
    )
    run = store.create_run(prepared)
    rows = tuple(_completed(c, (assessment("flag", "match", strength=1 + (i % 3)),), None) for i, c in enumerate(cands))
    store.commit_batch(run.run_id, batch_id="b1", results=rows, attempt=_attempt("b1", tuple(c.candidate_id for c in cands)))
    q = ResultQuery(
        run_id=run.run_id,
        page=PageRequest(page=1, page_size=5),
        sort=SortSpec(key="candidate_id", descending=False),
    )
    page1 = reader.query(q)
    assert len(page1.items) == 5
    assert page1.has_next is True
    assert page1.has_prev is False
    ids = reader.ordered_ids(q)
    first = ids[0]
    last_on_page1 = ids[4]
    pos = reader.neighbor(q, last_on_page1)
    assert pos is not None
    assert pos.next_id == ids[5]
    assert pos.position == 5
    assert pos.page == 1
    next_pos = reader.neighbor(q, pos.next_id)
    assert next_pos is not None and next_pos.page == 2
    assert reconcile_selected_candidate(last_on_page1, ids) == last_on_page1
    assert reconcile_selected_candidate("gone", ids) == first
    assert reconcile_selected_candidate("gone", ()) is None
    q2 = ResultQuery(
        run_id=run.run_id,
        page=PageRequest(page=2, page_size=5),
        sort=SortSpec(key="candidate_id", descending=False),
    )
    page2 = reader.query(q2)
    assert page2.items[0].candidate_id == ids[5]


def test_review_supersession_and_conflict_guard(tmp_path: Path) -> None:
    store, reader, reviews = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan()
    cand = prepared.candidates[0]
    run = store.create_run(prepared)
    store.commit_batch(
        run.run_id,
        batch_id="b1",
        results=(_completed(cand, (assessment("flag", "match", strength=2),), None),),
        attempt=_attempt("b1", (cand.candidate_id,)),
    )
    first = reviews.add_review(run.run_id, cand.candidate_id, judgment="unsure", note="v1", setup_id="flag")
    second = reviews.add_review(run.run_id, cand.candidate_id, judgment="agree", note="v2", setup_id="flag")
    assert second.supersedes_review_id == first.review_id
    assert reviews.current_review(run.run_id, cand.candidate_id, setup_id="flag").judgment == "agree"
    history = reviews.list_reviews(run.run_id, cand.candidate_id)
    assert history[0].note == "v1"
    assert history[0].judgment == "unsure"
    with pytest.raises(VisionError, match="conflicting review"):
        reviews.add_review(
            run.run_id,
            cand.candidate_id,
            judgment="disagree",
            setup_id="flag",
            expected_current_id=first.review_id,
        )
    labels = Path(__file__).resolve().parents[1] / "labels.csv"
    before = labels.read_bytes() if labels.exists() else b""
    reviews.add_review(run.run_id, cand.candidate_id, judgment="disagree", note="reject flag", setup_id="flag")
    if labels.exists():
        assert labels.read_bytes() == before
    row = reader.query(ResultQuery(run_id=run.run_id)).items[0]
    assert row.review_state == "disagree"
    export = reader.export_query(ResultQuery(run_id=run.run_id, page=PageRequest(page=1, page_size=1)))
    assert len(export.candidates) == 1
    assert export.assessments[0]["review_judgment"] == "disagree"


def test_skips_errors_unknown_diagnostics_and_missing_chart(tmp_path: Path) -> None:
    store, reader, _ = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan(
        skips=(
            InputSkip(ticker="SHORT", kind="short_window", message="too few bars", bar_count=3),
            InputSkip(ticker="STALE", kind="stale", message="behind expected session"),
        ),
        diagnostics=ScanDiagnostics(universe_count=9, not_eligible_count=None, unavailable=("not_eligible_count",)),
    )
    run = store.create_run(prepared)
    cand = prepared.candidates[0]
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
                error=AttemptError(kind="timeout", message="timed out", retryable=True),
                attempt_id=None,
                source_digest=cand.source_digest,
            ),
            CandidateResult(
                candidate_id="skip:SHORT:short_window",
                status="skipped",
                ticker="SHORT",
                asof_date=cand.asof_date,
                features=cand.features,
                eligible_setup_ids=(),
                assessments=(),
                artifact_id=None,
                error=AttemptError(kind="short_window", message="too few bars", retryable=False),
                attempt_id=None,
                source_digest="",
            ),
        ],
    )
    frozen = reader.load_frozen_inputs(run.run_id)
    assert frozen.diagnostics.not_eligible_count is None
    errors = reader.query(ResultQuery(run_id=run.run_id), statuses=("error", "skipped"))
    assert {r.status for r in errors.items} == {"error", "skipped"}
    detail = reader.get_detail(run.run_id, cand.candidate_id)
    assert detail.row.chart_ref is None
    assert detail.row.error.kind == "timeout"
    with pytest.raises(VisionError):
        reader.get_artifact_bytes(run.run_id, "nope")


def test_filesystem_store_with_scanner_fake_classifier(tmp_path: Path) -> None:
    from screener_loader.vision.scan import Scanner

    store = FilesystemRunStore(tmp_path)
    prepared = sample_prepared_scan(mode="demo")
    cid = prepared.candidates[0].candidate_id
    classifier = FakeClassifier(script={cid: (assessment("flag", "match", strength=2),)})
    scanner = Scanner(renderer=FakeRenderer(), compiler=FakeCompiler(), classifier=classifier, store=store)
    outcome = scanner.run(prepared)
    assert outcome.status in {"completed", "partial"}
    assert classifier.calls >= 1
    rows = store.list_candidate_results(outcome.run_id)
    completed = [r for r in rows if r.status == "completed"]
    assert completed
    ref = store.find_artifact_for_candidate(outcome.run_id, completed[0].candidate_id)
    assert ref is not None
    png = store.get_artifact_bytes(outcome.run_id, ref.artifact_id)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    calls_after = classifier.calls
    reader = FilesystemResultReader(tmp_path, store=store, reviews=FilesystemReviewStore(tmp_path, store=store))
    reader.query(ResultQuery(run_id=outcome.run_id))
    reader.get_detail(outcome.run_id, cid)
    assert classifier.calls == calls_after


def test_seeded_demo_store_has_cross_page_and_multi_flag(tmp_path: Path) -> None:
    store, reader, reviews = seed_synthetic_runs(tmp_path)
    runs = reader.list_runs()
    assert len(runs) >= 5
    mains = [r for r in runs if r.status == "partial" and r.config.mode == "demo"]
    main = max(mains, key=lambda r: reader.counts(r.run_id).candidates)
    q = ResultQuery(run_id=main.run_id, verdicts=("match",), page=PageRequest(page=1, page_size=10))
    page = reader.query(q)
    assert page.total_candidates > 10
    assert page.has_next
    aapl = next(
        r for r in reader.query(ResultQuery(run_id=main.run_id, page=PageRequest(1, 50))).items if r.ticker == "AAPL"
    )
    assert len(aapl.matched_setups) == 2
    assert reviews.current_review(main.run_id, aapl.candidate_id, setup_id="flag").judgment == "agree"
    dry = next(r for r in runs if r.status == "dry_run")
    dry_page = reader.query(ResultQuery(run_id=dry.run_id))
    assert dry_page.items
    assert all(item.status == "pending" for item in dry_page.items)
    empty = next(r for r in runs if reader.counts(r.run_id).candidates == 0)
    assert reader.query(ResultQuery(run_id=empty.run_id)).total_candidates == 0
    export = reader.export_query(q)
    assert len(export.candidates) == page.total_candidates
    assert len(export.candidates) > len(page.items)
    assert store.load_run(main.run_id).synthetic is True
