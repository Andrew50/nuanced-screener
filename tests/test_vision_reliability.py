"""Regression tests for vision MVP correctness/reliability defects."""

from __future__ import annotations

import json
import random as random_mod
import time
from datetime import date
from types import SimpleNamespace
from threading import Event

import pytest

from screener_loader.vision.client import OpenAIClassifier, _collect_output_text
from screener_loader.vision.prompts import (
    INSTRUCTIONS,
    SnapshotRequestCompiler,
    estimate_output_tokens,
)
from screener_loader.vision.scan import Scanner
from screener_loader.vision.snapshots import freeze_prepared_scan, make_candidate_input
from screener_loader.vision.store import FilesystemRunStore, aggregate_attempt_usage
from screener_loader.vision.types import (
    AttemptError,
    FeatureValue,
    InputSkip,
    OversizeRequestError,
    ScanConfig,
    TokenUsage,
)
from vision_support import (
    FakeClassifier,
    FakeClock,
    FakeCompiler,
    FakeRenderer,
    FakeRng,
    InMemoryRunStore,
    default_profile,
    feature_value_from_mapping,
    make_example,
    make_spec,
    sample_prepared_scan,
    snapshot_pair,
    window_for,
)


def _scan(classifier=None, compiler=None, store=None, **kwargs):
    clock = kwargs.pop("clock", FakeClock())
    rng = kwargs.pop("rng", FakeRng())
    return Scanner(
        renderer=FakeRenderer(),
        compiler=compiler or FakeCompiler(),
        classifier=classifier or FakeClassifier(),
        store=store or InMemoryRunStore(),
        sleep=kwargs.pop("sleep", clock.sleep),
        random=kwargs.pop("random", rng.random),
        **kwargs,
    )


def _batch_context():
    prepared = sample_prepared_scan()
    renderer = FakeRenderer()
    examples = []
    for ex in prepared.examples:
        if ex.type == "image":
            examples.append(renderer.wrap_upload(ex))
        else:
            examples.append(renderer.render(ex.window, prepared.profile, title=ex.scoped_id))
            from dataclasses import replace

            examples[-1] = replace(
                examples[-1], kind="example", setup_id=ex.setup_id, example_id=ex.example_id
            )
    from dataclasses import replace as _replace

    cand_arts = [
        _replace(renderer.render(c.window, prepared.profile, title=c.ticker), kind="candidate", candidate_id=c.candidate_id)
        for c in prepared.candidates
    ]
    req = SnapshotRequestCompiler().compile(
        setups=prepared.setups,
        example_artifacts=examples,
        examples=prepared.examples,
        candidate_artifacts=cand_arts,
        candidates=prepared.candidates,
        config=prepared.config,
        batch_id="b-test",
    )
    return prepared, req, cand_arts, examples


def _two_candidates(*, batch_size: int = 1, max_retries: int = 0, **config_kw):
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    cands = []
    for ticker, close in (("AAA", 10.0), ("BBB", 50.0)):
        cands.append(
            make_candidate_input(
                ticker=ticker,
                window=window_for(ticker, n=2),
                features=feature_value_from_mapping(
                    {"close": close, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04}
                ),
                eligible_setup_ids=("flag",),
                profile=profile,
            )
        )
    return freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(
            model="gpt-4.1-2025-04-14",
            batch_size=batch_size,
            max_concurrency=1,
            max_retries=max_retries,
            **config_kw,
        ),
    )


def _sdk_response(text: str, *, refusal: str | None = None, status: str = "completed"):
    pytest.importorskip("openai")
    from openai.types.responses.response import Response

    if refusal:
        content = [{"type": "refusal", "refusal": refusal}]
    else:
        content = [{"type": "output_text", "text": text, "annotations": []}]
    payload = {
        "id": "resp_local",
        "object": "response",
        "created_at": 1_719_000_000,
        "status": status,
        "model": "gpt-4.1-2025-04-14",
        "output": [
            {
                "id": "msg_local",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": content,
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    try:
        response = Response.model_validate(payload)
    except Exception:
        response = Response.construct(**payload)
    assert isinstance(response.output_text, str)
    if not refusal:
        # Real SDK output_text re-aggregates the same blocks. Parsing both would duplicate JSON.
        assert response.output_text == text
        assert response.output
    return response


def test_collect_output_text_reads_sdk_response_once() -> None:
    body = {
        "results": [
            {
                "candidate_id": "AAPL:2026-06-18:1d:100",
                "assessments": [
                    {
                        "setup_id": "flag",
                        "verdict": "no_match",
                        "match_strength": None,
                        "reason": "No coil.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    }
                ],
            }
        ]
    }
    text = json.dumps(body)
    response = _sdk_response(text)
    got, refusals = _collect_output_text(response)
    assert refusals == ()
    assert got == text
    assert got.count('"results"') == 1
    parsed = json.loads(got)
    assert parsed == body

    refused = _sdk_response("", refusal="policy refusal")
    _text, refusals = _collect_output_text(refused)
    assert refusals
    assert "policy" in refusals[0]


def test_openai_classifier_parses_sdk_response_without_duplication() -> None:
    _prepared, req, cand_arts, examples = _batch_context()
    images = {a.artifact_id: a.image.png_bytes for a in list(cand_arts) + list(examples)}
    cid = req.candidate_ids[0]
    body = {
        "results": [
            {
                "candidate_id": cid,
                "assessments": [
                    {
                        "setup_id": "flag",
                        "verdict": "no_match",
                        "match_strength": None,
                        "reason": "No coil.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    }
                ],
            }
        ]
    }
    text = json.dumps(body)

    class Responses:
        def create(self, **kwargs):
            return _sdk_response(text)

    attempt = OpenAIClassifier(client=SimpleNamespace(responses=Responses()), images=images).classify(req)
    assert attempt.error is None
    assert attempt.sanitized_output == body
    assert json.dumps(attempt.sanitized_output).count('"results"') == 1

    class RefusalResponses:
        def create(self, **kwargs):
            return _sdk_response("", refusal="not allowed")

    refused = OpenAIClassifier(client=SimpleNamespace(responses=RefusalResponses()), images=images).classify(req)
    assert refused.error is not None
    assert refused.error.kind == "refusal"


def test_production_retry_timing_and_injected_sleeper() -> None:
    production = Scanner(
        renderer=FakeRenderer(),
        compiler=FakeCompiler(),
        classifier=FakeClassifier(),
        store=InMemoryRunStore(),
    )
    assert production._sleep is time.sleep
    assert production._random is random_mod.random

    prepared = sample_prepared_scan()
    err = AttemptError(kind="rate_limit", message="wait", retryable=True, retry_after_seconds=2.0)
    clf = FakeClassifier()
    clf.fail_next(1, err)
    clock = FakeClock()
    rng = FakeRng([0.0])
    one = freeze_prepared_scan(
        profile=prepared.profile,
        setups=prepared.setups,
        examples=prepared.examples,
        candidates=prepared.candidates,
        config=ScanConfig(model=prepared.config.model, max_retries=2, jitter_ratio=0.0),
    )
    out = _scan(classifier=clf, clock=clock, rng=rng).run(one)
    assert out.status == "completed"
    assert clock.sleeps
    assert clock.sleeps[0] == pytest.approx(2.0)

    err2 = AttemptError(kind="rate_limit", message="backoff", retryable=True)
    clf2 = FakeClassifier()
    clf2.fail_next(2, err2)
    clock2 = FakeClock()
    cfg = ScanConfig(
        model=prepared.config.model,
        max_retries=3,
        retry_backoff_seconds=1.0,
        retry_backoff_max_seconds=20.0,
        jitter_ratio=0.0,
    )
    two = freeze_prepared_scan(
        profile=prepared.profile,
        setups=prepared.setups,
        examples=prepared.examples,
        candidates=prepared.candidates,
        config=cfg,
    )
    out2 = _scan(classifier=clf2, clock=clock2, rng=FakeRng([0.0])).run(two)
    assert out2.status == "completed"
    assert clock2.sleeps[:2] == pytest.approx([1.0, 2.0])


def test_resume_status_from_persisted_outcomes(tmp_path) -> None:
    prepared = _two_candidates(batch_size=1, max_retries=0)
    store = FilesystemRunStore(tmp_path)

    # 1. all-success
    ok = FakeClassifier()
    out = _scan(classifier=ok, store=store).run(prepared)
    assert out.status == "completed"
    assert out.summary.candidates_completed == 2
    assert out.summary.candidates_error == 0
    assert {r.status for r in store.list_candidate_results(out.run_id) if not r.candidate_id.startswith("skip:")} == {
        "completed"
    }

    # 2. partial then resume
    cancel = Event()

    class Once(FakeClassifier):
        def classify(self, request):
            result = super().classify(request)
            cancel.set()
            return result

    store2 = FilesystemRunStore(tmp_path / "partial")
    once = Once()
    partial_scan = _scan(classifier=once, store=store2)
    partial = partial_scan.run(prepared, cancel=cancel)
    assert partial.status == "cancelled"
    completed_ids = store2.list_committed_candidate_ids(partial.run_id)
    assert len(completed_ids) == 1
    calls_after_cancel = once.calls
    resumed = partial_scan.resume(partial.run_id)
    assert resumed.status == "completed"
    assert store2.list_committed_candidate_ids(partial.run_id) == {
        c.candidate_id for c in prepared.candidates
    }
    later = once.requests[calls_after_cancel:]
    assert not any(set(req.candidate_ids) <= completed_ids and completed_ids for req in later)

    # 3+4. error run resumes; still-failing errors stay failed, not completed
    store3 = FilesystemRunStore(tmp_path / "errors")
    boom = AttemptError(kind="provider", message="boom", retryable=False)
    failing = FakeClassifier(error=boom)
    err_run = _scan(classifier=failing, store=store3).run(prepared)
    assert err_run.status == "failed"
    assert err_run.summary.candidates_error == 2
    assert err_run.summary.candidates_completed == 0
    first_calls = failing.calls
    still = _scan(classifier=failing, store=store3).resume(err_run.run_id)
    assert still.status == "failed"
    assert still.summary.candidates_error == 2
    assert still.summary.candidates_completed == 0
    assert failing.calls > first_calls
    rows = [r for r in store3.list_candidate_results(err_run.run_id) if r.status == "error"]
    assert len(rows) == 2

    # 5+6. retrying errors can complete without duplicating successes
    failing.error = None
    recovered = _scan(classifier=failing, store=store3).resume(err_run.run_id)
    assert recovered.status == "completed"
    assert recovered.summary.candidates_completed == 2
    assert recovered.summary.candidates_error == 0
    completed = [r for r in store3.list_candidate_results(err_run.run_id) if r.status == "completed"]
    assert len(completed) == 2
    assert len({r.candidate_id for r in completed}) == 2


def test_usage_aggregated_once_per_attempt_id(tmp_path) -> None:
    prepared = sample_prepared_scan()
    store = FilesystemRunStore(tmp_path)
    clf = FakeClassifier()
    out = _scan(classifier=clf, store=store).run(prepared)
    attempts = store.list_attempts(out.run_id)
    assert len(attempts) >= 2  # raw classify + validated commit
    ids = {a.attempt_id for a in attempts}
    assert len(ids) == 1
    count, usage = aggregate_attempt_usage(attempts)
    assert count == 1
    assert usage == TokenUsage(input_tokens=20, output_tokens=40, total_tokens=60)
    manifest = store.export_manifest(out.run_id)
    assert manifest["attempts"] == 1
    assert manifest["usage"]["total_tokens"] == 60
    assert out.summary.attempts == 1
    assert out.summary.usage.total_tokens == 60

    # resume/reload must not inflate
    again = _scan(classifier=clf, store=store).resume(out.run_id)
    assert again.summary.attempts == 1
    assert store.export_manifest(out.run_id)["attempts"] == 1

    # retries are distinct attempt ids
    err = AttemptError(kind="rate_limit", message="wait", retryable=True, retry_after_seconds=0.0)
    clf2 = FakeClassifier()
    clf2.fail_next(1, err)
    store2 = FilesystemRunStore(tmp_path / "retry")
    out2 = _scan(classifier=clf2, store=store2, clock=FakeClock()).run(prepared)
    count2, usage2 = aggregate_attempt_usage(store2.list_attempts(out2.run_id))
    assert count2 == 2
    assert usage2.total_tokens == 10 + 60
    assert store2.export_manifest(out2.run_id)["attempts"] == 2


def test_skipped_metadata_and_count_reconciliation() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    aapl = make_candidate_input(
        ticker="AAPL",
        window=window_for("AAPL", n=2),
        features=feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1_000.0, "adr_pct_20": 0.04}),
        eligible_setup_ids=("flag",),
        profile=profile,
    )
    skip = InputSkip(
        ticker="MSFT",
        kind="short_window",
        message="too few bars",
        asof_date=date(2026, 6, 10),
        bar_count=3,
        features=FeatureValue(close=77.0, dollar_vol_avg_20=9_000.0, adr_pct_20=0.12),
    )
    missing = InputSkip(ticker="ZZZ", kind="unavailable_input", message="no rows")
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=(aapl,),
        skips=(skip, missing),
        config=ScanConfig(model="gpt-4.1-2025-04-14", mode="demo"),
    )
    store = InMemoryRunStore()
    out = _scan(store=store).run(prepared)
    rows = {r.ticker: r for r in store.list_candidate_results(out.run_id)}
    assert rows["MSFT"].features.close == 77.0
    assert rows["MSFT"].features.close != aapl.features.close
    assert rows["ZZZ"].features.close is None
    assert rows["ZZZ"].features.dollar_vol_avg_20 is None
    summary = out.summary
    assert summary.candidates_completed == 1
    assert summary.candidates_skipped == 2
    assert summary.candidates_error == 0
    assert summary.candidates_pending == 0
    assert summary.candidates_total == 3
    assert (
        summary.candidates_completed
        + summary.candidates_error
        + summary.candidates_skipped
        + summary.candidates_pending
        == summary.candidates_total
    )


def test_resume_uses_frozen_compiler_instructions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from screener_loader.vision import prompts as prompts_mod

    prepared = _two_candidates(batch_size=1, max_retries=0)
    assert prepared.compiler is not None
    original = prepared.compiler.instructions
    assert original == INSTRUCTIONS
    store = FilesystemRunStore(tmp_path)
    cancel = Event()

    class Once(FakeClassifier):
        def classify(self, request):
            result = super().classify(request)
            cancel.set()
            return result

    clf = Once()
    scanner = Scanner(
        renderer=FakeRenderer(),
        compiler=SnapshotRequestCompiler(),
        classifier=clf,
        store=store,
        sleep=lambda _s: None,
        random=lambda: 0.0,
    )
    out = scanner.run(prepared, cancel=cancel)
    assert out.status == "cancelled"
    monkeypatch.setattr(
        prompts_mod,
        "INSTRUCTIONS",
        "CHANGED INSTRUCTIONS — resume must not use this text.",
    )
    resumed_compiler = SnapshotRequestCompiler()
    assert "CHANGED INSTRUCTIONS" in resumed_compiler.instructions
    scanner2 = Scanner(
        renderer=FakeRenderer(),
        compiler=resumed_compiler,
        classifier=clf,
        store=store,
        sleep=lambda _s: None,
        random=lambda: 0.0,
    )
    scanner2.resume(out.run_id)
    later = [req for req in clf.requests if "CHANGED INSTRUCTIONS" in (req.blocks[0].text or "")]
    assert later == []
    frozen = store.load_frozen_inputs(out.run_id)
    assert frozen.compiler is not None
    assert frozen.compiler.instructions == original
    assert any(original[:40] in (req.blocks[0].text or "") for req in clf.requests)


def test_output_token_budget_enforced_and_split() -> None:
    spec = make_spec("flag", lookback_bars=2)
    setup, examples = snapshot_pair(spec, [make_example("ex_canonical")])
    profile = default_profile(lookback_bars=2)
    cands = [
        make_candidate_input(
            ticker=t,
            window=window_for(t, n=2),
            features=feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04}),
            eligible_setup_ids=("flag",),
            profile=profile,
        )
        for t in ("AAA", "BBB")
    ]
    one = estimate_output_tokens(1)
    two = estimate_output_tokens(2)
    assert two > one
    prepared = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=tuple(cands),
        config=ScanConfig(
            model="gpt-4.1-2025-04-14",
            batch_size=2,
            max_concurrency=1,
            max_output_tokens=one + 10,
            max_images_per_request=24,
        ),
    )
    compiler = SnapshotRequestCompiler()
    clf = FakeClassifier()
    out = _scan(classifier=clf, compiler=compiler).run(prepared)
    assert out.status == "completed"
    assert clf.calls == 2
    assert all(len(req.candidate_ids) == 1 for req in clf.requests)

    too_small = freeze_prepared_scan(
        profile=profile,
        setups=(setup,),
        examples=examples,
        candidates=(cands[0],),
        config=ScanConfig(model="gpt-4.1-2025-04-14", max_output_tokens=one - 1),
    )
    with pytest.raises(OversizeRequestError, match="alone is estimated"):
        _scan(compiler=compiler).run(too_small)


def test_reference_image_limits() -> None:
    prepared, req, cand_arts, examples = _batch_context()
    compiler = SnapshotRequestCompiler()
    # sample scan: 4 reference images + 1 candidate
    exact = ScanConfig(model=prepared.config.model, max_images_per_request=req.estimates.total_image_count)
    ok = compiler.compile(
        setups=prepared.setups,
        example_artifacts=examples,
        examples=prepared.examples,
        candidate_artifacts=cand_arts,
        candidates=prepared.candidates,
        config=exact,
        batch_id="at-limit",
    )
    assert ok.estimates.total_image_count == exact.max_images_per_request

    cand_over = ScanConfig(model=prepared.config.model, max_images_per_request=req.estimates.reference_image_count)
    with pytest.raises(OversizeRequestError, match="images"):
        compiler.compile(
            setups=prepared.setups,
            example_artifacts=examples,
            examples=prepared.examples,
            candidate_artifacts=cand_arts,
            candidates=prepared.candidates,
            config=cand_over,
            batch_id="cand-over",
        )

    ref_only = ScanConfig(model=prepared.config.model, max_images_per_request=1)
    with pytest.raises(OversizeRequestError, match="Reference images"):
        _scan(compiler=compiler).run(
            freeze_prepared_scan(
                profile=prepared.profile,
                setups=prepared.setups,
                examples=prepared.examples,
                candidates=prepared.candidates,
                config=ref_only,
            )
        )
