from __future__ import annotations

from types import SimpleNamespace

import pytest

from screener_loader.vision.client import OpenAIClassifier
from screener_loader.vision.prompts import SnapshotRequestCompiler
from screener_loader.vision.types import ScanConfig
from screener_loader.vision.validation import SemanticValidationError, validate_classification_payload
from vision_support import (
    FakeRenderer,
    make_example,
    make_spec,
    sample_prepared_scan,
    snapshot_pair,
    window_for,
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
    cand_arts = [
        replace_cand(renderer.render(c.window, prepared.profile, title=c.ticker), c.candidate_id)
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


def replace_cand(art, candidate_id: str):
    from dataclasses import replace

    return replace(art, kind="candidate", candidate_id=candidate_id)


def test_vision_compiler_orders_setups_examples_candidates_and_keeps_all_examples() -> None:
    prepared, req, _, _ = _batch_context()
    purposes = [b.purpose for b in req.blocks]
    assert purposes[0] == "instructions"
    assert "setup" in purposes
    assert purposes.index("setup") < next(i for i, p in enumerate(purposes) if p == "example")
    assert purposes.index("example") < next(i for i, p in enumerate(purposes) if p == "candidate")
    example_texts = [b.text for b in req.blocks if b.kind == "text" and b.purpose == "example"]
    assert any("ex_canonical" in (t or "") for t in example_texts)
    assert any("ex_near" in (t or "") for t in example_texts)
    assert any("upload_1" in (t or "") for t in example_texts)
    assert req.estimates.notes.lower().startswith("all")
    assert "estimate" in req.estimates.notes.lower()
    assert req.candidate_ids == tuple(c.candidate_id for c in prepared.candidates)
    joined = "\n".join(b.text or "" for b in req.blocks if b.kind == "text")
    assert "USD/share" in joined
    assert "not an instruction source" in joined
    assert "compile_prompt" not in joined
    assert req.model == prepared.config.model


def test_vision_compiler_oversize_does_not_drop_examples() -> None:
    prepared, _, cand_arts, examples = _batch_context()
    cfg = ScanConfig(model=prepared.config.model, max_images_per_request=1)
    with pytest.raises(Exception) as exc:
        SnapshotRequestCompiler().compile(
            setups=prepared.setups,
            example_artifacts=examples,
            examples=prepared.examples,
            candidate_artifacts=cand_arts,
            candidates=prepared.candidates,
            config=cfg,
            batch_id="b-over",
        )
    assert "not dropped" in str(exc.value).lower() or "max_images" in str(exc.value)


def _payload(cid: str, assessments: list[dict]) -> dict:
    return {"results": [{"candidate_id": cid, "assessments": assessments}]}


def test_vision_validation_reordered_complete_coverage() -> None:
    prepared = sample_prepared_scan()
    cid = prepared.candidates[0].candidate_id
    row = {
        "setup_id": "flag",
        "verdict": "match",
        "match_strength": 2,
        "reason": "Tight coil under a clear shelf with drying volume.",
        "violated_required_rule_ids": [],
        "missing_evidence": [],
    }
    out = validate_classification_payload(
        {"results": [{"candidate_id": cid, "assessments": [row]}]},
        candidates=prepared.candidates,
        setups=prepared.setups,
    )
    assert out[0].assessments[0].verdict == "match"
    assert out[0].assessments[0].match_strength == 2


@pytest.mark.parametrize(
    "payload_mut, fragment",
    [
        (lambda cid: {"results": []}, "missing_candidates"),
        (
            lambda cid: _payload(
                cid,
                [
                    {
                        "setup_id": "flag",
                        "verdict": "no_match",
                        "match_strength": None,
                        "reason": "Choppy.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    },
                    {
                        "setup_id": "flag",
                        "verdict": "match",
                        "match_strength": 1,
                        "reason": "Dup.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    },
                ],
            ),
            "duplicate_setup",
        ),
        (
            lambda cid: _payload(
                cid,
                [
                    {
                        "setup_id": "nope",
                        "verdict": "no_match",
                        "match_strength": None,
                        "reason": "x",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    }
                ],
            ),
            "unknown_setup",
        ),
        (
            lambda cid: {
                "results": [
                    {
                        "candidate_id": cid,
                        "assessments": [
                            {
                                "setup_id": "flag",
                                "verdict": "match",
                                "match_strength": 2,
                                "reason": "ok",
                                "violated_required_rule_ids": [],
                                "missing_evidence": [],
                            }
                        ],
                    },
                    {
                        "candidate_id": cid,
                        "assessments": [
                            {
                                "setup_id": "flag",
                                "verdict": "no_match",
                                "match_strength": None,
                                "reason": "dup cid",
                                "violated_required_rule_ids": [],
                                "missing_evidence": [],
                            }
                        ],
                    },
                ]
            },
            "duplicate_candidate",
        ),
    ],
)
def test_vision_validation_rejects_bad_coverage(payload_mut, fragment) -> None:
    prepared = sample_prepared_scan()
    cid = prepared.candidates[0].candidate_id
    with pytest.raises(SemanticValidationError) as exc:
        validate_classification_payload(payload_mut(cid), candidates=prepared.candidates, setups=prepared.setups)
    assert fragment in ",".join(exc.value.issues)


def test_vision_validation_ineligible_and_multiple_matches_and_uncertain() -> None:
    flag = make_spec("flag")
    ep = make_spec("ep", required=("gap",))
    flag_s, _ = snapshot_pair(flag, [make_example("a")])
    ep_s, _ = snapshot_pair(ep, [make_example("a")])
    from screener_loader.vision.snapshots import make_candidate_input as mci
    from vision_support import default_profile, feature_value_from_mapping

    profile = default_profile()
    win = window_for("NVDA", n=20)
    cand = mci(
        ticker="NVDA",
        window=win,
        features=feature_value_from_mapping({"close": 10.0, "dollar_vol_avg_20": 1.0, "adr_pct_20": 0.04}),
        eligible_setup_ids=("flag",),
        profile=profile,
    )
    payload = {
        "results": [
            {
                "candidate_id": cand.candidate_id,
                "assessments": [
                    {
                        "setup_id": "ep",
                        "verdict": "match",
                        "match_strength": 3,
                        "reason": "Should not grade an ineligible setup.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": [],
                    }
                ],
            }
        ]
    }
    with pytest.raises(SemanticValidationError):
        validate_classification_payload(payload, candidates=(cand,), setups=(flag_s, ep_s))

    multi = {
        "results": [
            {
                "candidate_id": cand.candidate_id,
                "assessments": [
                    {
                        "setup_id": "flag",
                        "verdict": "uncertain",
                        "match_strength": None,
                        "reason": "Right edge is off-screen so the breakout is not visible.",
                        "violated_required_rule_ids": [],
                        "missing_evidence": ["right-edge resolution"],
                    }
                ],
            }
        ]
    }
    out = validate_classification_payload(multi, candidates=(cand,), setups=(flag_s, ep_s))
    assert out[0].assessments[0].verdict == "uncertain"


def test_vision_client_one_attempt_no_sdk_retries_and_local_images() -> None:
    prepared, req, cand_arts, examples = _batch_context()
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
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_1",
                status="completed",
                output_text=__import__("json").dumps(body),
                output=[],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
                incomplete_details=None,
            )

    client = SimpleNamespace(responses=Responses(), max_retries=0)
    clf = OpenAIClassifier(client=client, images=images)
    attempt = clf.classify(req)
    assert attempt.error is None
    assert attempt.response_id == "resp_1"
    assert attempt.usage.input_tokens == 11
    assert captured["model"] == req.model
    assert "temperature" not in captured
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["strict"] is True
    content = captured["input"][0]["content"]
    assert any(p["type"] == "input_image" and p["image_url"].startswith("data:image/png;base64,") for p in content)
    assert client.max_retries == 0


def test_vision_client_refusal_truncation_rate_limit() -> None:
    prepared, req, cand_arts, examples = _batch_context()
    images = {a.artifact_id: a.image.png_bytes for a in list(cand_arts) + list(examples)}

    class RefusalResponses:
        def create(self, **kwargs):
            part = SimpleNamespace(type="refusal", refusal="policy")
            msg = SimpleNamespace(type="message", content=[part])
            return SimpleNamespace(id="r", status="completed", output_text="", output=[msg], usage=None, incomplete_details=None)

    attempt = OpenAIClassifier(client=SimpleNamespace(responses=RefusalResponses()), images=images).classify(req)
    assert attempt.error.kind == "refusal"
    assert attempt.accepted is False

    class Incomplete:
        def create(self, **kwargs):
            return SimpleNamespace(
                id="r2",
                status="incomplete",
                output_text="",
                output=[],
                usage=SimpleNamespace(input_tokens=3, output_tokens=1, total_tokens=4),
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )

    inc = OpenAIClassifier(client=SimpleNamespace(responses=Incomplete()), images=images).classify(req)
    assert inc.error.kind == "incomplete"
    assert inc.usage.output_tokens == 1

    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "1.5"})

    class RL:
        def create(self, **kwargs):
            raise RateLimitError("slow down")

    rl = OpenAIClassifier(client=SimpleNamespace(responses=RL()), images=images).classify(req)
    assert rl.error.kind == "rate_limit"
    assert rl.error.retry_after_seconds == 1.5
    assert rl.error.retryable is True

    class Quota:
        def create(self, **kwargs):
            raise RateLimitError(
                "Error code: 429 - {'error': {'code': 'credit_balance_exhausted', 'type': 'insufficient_quota'}}"
            )

    quota = OpenAIClassifier(client=SimpleNamespace(responses=Quota()), images=images).classify(req)
    assert quota.error.kind == "config"
    assert quota.error.retryable is False
    assert "no credits" in quota.error.message
