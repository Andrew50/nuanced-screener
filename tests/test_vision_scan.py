from __future__ import annotations

from datetime import date
from threading import Event

import pytest

from screener_loader.vision.scan import Scanner
from screener_loader.vision.snapshots import freeze_prepared_scan, make_candidate_input
from screener_loader.vision.types import (
    AttemptError,
    FrozenInputConflictError,
    ScanConfig,
    ScanDiagnostics,
)
from vision_support import (
    FakeClassifier,
    FakeClock,
    FakeCompiler,
    FakeRenderer,
    FakeRng,
    InMemoryRunStore,
    assessment,
    default_profile,
    feature_value_from_mapping,
    make_example,
    make_spec,
    sample_prepared_scan,
    snapshot_pair,
    window_for,
)


def _scanner(classifier=None, compiler=None, store=None, **kwargs):
    clock = kwargs.pop("clock", FakeClock())
    rng = kwargs.pop("rng", FakeRng())
    return Scanner(
        renderer=FakeRenderer(),
        compiler=compiler or FakeCompiler(),
        classifier=classifier or FakeClassifier(),
        store=store or InMemoryRunStore(),
        sleep=clock.sleep,
        random=rng.random,
        **kwargs,
    )


def test_vision_scan_complete_fake_and_zero_candidates() -> None:
    prepared = sample_prepared_scan()
    cid = prepared.candidates[0].candidate_id
    clf = FakeClassifier(script={cid: (assessment("flag", "match", strength=2),)})
    store = InMemoryRunStore()
    scanner = _scanner(classifier=clf, store=store)
    out = scanner.run(prepared)
    assert out.status == "completed"
    assert out.summary.candidates_completed == 1
    assert out.summary.setup_matches == 1
    assert clf.calls == 1
    row = store.list_candidate_results(out.run_id)[0]
    assert row.assessments[0].verdict == "match"
    png = store.get_artifact_bytes(out.run_id, next(iter(store.artifacts))[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    empty = freeze_prepared_scan(
        profile=prepared.profile,
        setups=prepared.setups,
        examples=prepared.examples,
        candidates=(),
        config=prepared.config,
        diagnostics=ScanDiagnostics(eligible_union_count=0),
    )
    out0 = _scanner().run(empty)
    assert out0.status == "completed"
    assert out0.summary.candidates_total == 0


def test_vision_scan_dry_run_makes_zero_classifier_calls() -> None:
    prepared = sample_prepared_scan(mode="dry_run")
    clf = FakeClassifier()
    compiler = FakeCompiler()
    store = InMemoryRunStore()
    out = _scanner(classifier=clf, compiler=compiler, store=store).run(prepared)
    assert out.status == "dry_run"
    assert clf.calls == 0
    assert compiler.calls >= 1
    assert out.summary.synthetic is False or prepared.config.mode == "dry_run"
    rows = store.list_candidate_results(out.run_id)
    assert rows and all(r.status == "completed" for r in rows)
    assert all(r.artifact_id for r in rows)
    assert all(not r.assessments for r in rows)


def test_vision_scan_partial_failure_retry_and_oversize_split() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    cands = []
    for i, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
        cands.append(
            make_candidate_input(
                ticker=ticker,
                window=window_for(ticker, n=2, asof=date(2026, 6, 18)),
                features=feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04}),
                eligible_setup_ids=("flag",),
                profile=profile,
            )
        )
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(model="gpt-4.1-2025-04-14", batch_size=4, max_concurrency=2, max_retries=2),
    )
    compiler = FakeCompiler()
    compiler.oversize_when_gt = 2
    clf = FakeClassifier()
    clock = FakeClock()
    scanner = _scanner(classifier=clf, compiler=compiler, clock=clock)
    out = scanner.run(prepared)
    assert out.status == "completed"
    assert out.summary.candidates_completed == 4
    assert compiler.max_seen_candidates <= 2 or compiler.calls >= 2

    err = AttemptError(kind="rate_limit", message="wait", retryable=True, retry_after_seconds=0.01)
    clf2 = FakeClassifier()
    clf2.fail_next(1, err)
    clock2 = FakeClock()
    one = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=(cands[0],),
        config=ScanConfig(model="gpt-4.1-2025-04-14", max_retries=2),
    )
    out2 = _scanner(classifier=clf2, clock=clock2).run(one)
    assert out2.status == "completed"
    assert clf2.calls >= 2
    assert clock2.sleeps


def test_vision_scan_cancellation_resume_dedup_and_changed_input() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    cands = [
        make_candidate_input(
            ticker=t,
            window=window_for(t, n=2),
            features=feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": None}),
            eligible_setup_ids=("flag",),
            profile=profile,
        )
        for t in ("AAA", "BBB", "CCC")
    ]
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(model="gpt-4.1-2025-04-14", batch_size=1, max_concurrency=1),
    )
    cancel = Event()

    class CancellingClassifier(FakeClassifier):
        def classify(self, request):
            cancel.set()
            return super().classify(request)

    store = InMemoryRunStore()
    cclf = CancellingClassifier()
    scanner = Scanner(renderer=FakeRenderer(), compiler=FakeCompiler(), classifier=cclf, store=store, sleep=lambda s: None, random=lambda: 0.0)
    out = scanner.run(prepared, cancel=cancel)
    assert out.status == "cancelled"
    first_calls = cclf.calls
    completed_after_cancel = set(store.list_committed_candidate_ids(out.run_id))
    scanner.resume(out.run_id)
    scanner.resume(out.run_id)
    later = cclf.requests[first_calls:]
    resubmitted = [req for req in later if completed_after_cancel and set(req.candidate_ids) <= completed_after_cancel]
    assert resubmitted == []
    other = sample_prepared_scan()
    with pytest.raises(FrozenInputConflictError):
        scanner.resume(out.run_id, other)


def test_vision_scan_bounded_inflight_3000_fake_workload() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    feats = feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04})
    cands = [
        make_candidate_input(
            ticker=f"T{i:04d}",
            window=window_for(f"T{i:04d}", n=2),
            features=feats,
            eligible_setup_ids=("flag",),
            profile=profile,
        )
        for i in range(3000)
    ]
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(
            model="gpt-4.1-2025-04-14",
            batch_size=10,
            max_concurrency=2,
            max_in_flight_batches=2,
            max_retries=0,
        ),
    )
    compiler = FakeCompiler()
    clf = FakeClassifier()
    scanner = _scanner(classifier=clf, compiler=compiler)
    out = scanner.run(prepared)
    assert out.status == "completed"
    assert out.summary.candidates_completed == 3000
    assert compiler.max_seen_candidates <= 10
    assert scanner.peak_in_flight <= 2
    assert clf.calls == 300


def test_vision_scan_durable_candidate_id_batch_order() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    # Insert in reverse ticker order; batches must still be candidate_id sorted.
    tickers = ["CCC", "AAA", "BBB"]
    cands = [
        make_candidate_input(
            ticker=t,
            window=window_for(t, n=2),
            features=feature_value_from_mapping({"close": 1.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04}),
            eligible_setup_ids=("flag",),
            profile=profile,
        )
        for t in tickers
    ]
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(model="gpt-4.1-2025-04-14", batch_size=2, max_concurrency=1),
    )
    clf = FakeClassifier()
    _scanner(classifier=clf).run(prepared)
    first_batch_ids = list(clf.requests[0].candidate_ids)
    assert first_batch_ids == sorted(first_batch_ids)


def test_vision_scan_auth_stops_dispatch() -> None:
    prepared = sample_prepared_scan()
    clf = FakeClassifier(error=AttemptError(kind="auth", message="bad key", retryable=False))
    out = _scanner(classifier=clf).run(prepared)
    assert out.status == "failed"
    assert clf.calls == 1
