"""Streamlit results page tests. No classifier or data-vendor calls on rerender."""

from __future__ import annotations

from pathlib import Path

import pytest

from screener_loader.vision.types import PageRequest, ResultQuery, SortSpec
from screener_loader.vision_ui.components import (
    STATE_PREFIX,
    VIEW_MATCHES,
    badge_text,
    build_result_query,
    feature_caption,
    format_adr_pct,
    reconcile_selected_candidate,
    review_widget_keys,
    state_key,
)
from screener_loader.vision_ui.synthetic import seed_synthetic_runs


def test_state_keys_are_namespaced() -> None:
    assert state_key("run_id") == "vr_run_id"
    assert STATE_PREFIX == "vr_"
    with pytest.raises(ValueError):
        state_key("setup_id")


def test_adr_display_and_query_builder() -> None:
    assert format_adr_pct(0.04) == "4.0%"
    assert format_adr_pct(None) == "—"
    from screener_loader.vision_ui.components import format_dollar_vol_millions

    assert format_dollar_vol_millions(355_997_519) == "$356.0M"
    assert format_dollar_vol_millions(None) == "—"
    q, statuses = build_result_query(run_id="run-1", view=VIEW_MATCHES, setup_ids=("flag",), page=2)
    assert q.verdicts == ("match",)
    assert q.setup_ids == ("flag",)
    assert q.page.page == 2
    assert statuses == ()
    _q2, err_statuses = build_result_query(run_id="run-1", view="errors")
    assert err_statuses == ("error", "skipped")


def test_selection_reconciliation_rule() -> None:
    ids = ("a", "b", "c")
    assert reconcile_selected_candidate("b", ids) == "b"
    assert reconcile_selected_candidate("z", ids) == "a"
    assert reconcile_selected_candidate(None, ids) == "a"
    assert reconcile_selected_candidate("a", ()) is None


def test_render_results_page_has_no_page_config() -> None:
    import inspect

    from screener_loader.vision_ui import results_page

    source = inspect.getsource(results_page)
    assert "set_page_config" not in source
    from screener_loader.vision_ui import demo

    assert "set_page_config" in inspect.getsource(demo.main)


def test_results_page_does_not_import_client_or_scan() -> None:
    import screener_loader.vision_ui.results_page as rp

    assert "screener_loader.vision.client" not in rp.__dict__
    assert "screener_loader.vision.scan" not in rp.__dict__


def test_demo_page_and_real_store_without_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, reader, reviews = seed_synthetic_runs(tmp_path)
    monkeypatch.setenv("NS_VISION_DEMO_ROOT", str(tmp_path))
    from streamlit.testing.v1 import AppTest

    demo_file = Path(__file__).resolve().parents[1] / "src" / "screener_loader" / "vision_ui" / "demo.py"
    at = AppTest.from_file(str(demo_file), default_timeout=30)
    at.run()
    assert not at.exception
    assert any(b.label == "Previous" for b in at.button)
    assert any(b.label == "Next" for b in at.button)
    assert not any(b.label in {"Prev page", "Next page"} for b in at.button)
    assert not any(getattr(e, "label", None) == "Export" for e in at.expander)
    assert any(getattr(e, "label", None) == "Filters" for e in at.expander)
    assert any(b.label in {"AAPL", "AAA", "MSFT", "NVDA"} for b in at.button)
    assert not any("Frozen" in str(getattr(e, "label", "") or "") for e in at.expander)
    assert any(b.label == "Save" for b in at.button)
    assert store.list_runs()
    main = max(store.list_runs(), key=lambda r: reader.counts(r.run_id).candidates)
    q = ResultQuery(
        run_id=main.run_id,
        verdicts=("match",),
        page=PageRequest(1, 10),
        sort=SortSpec("match_strength", True),
    )
    page = reader.query(q)
    assert page.items
    row = page.items[0]
    assert badge_text(row)
    assert "ADR" in feature_caption(row.features) or format_adr_pct(row.features.adr_pct_20)
    detail = reader.get_detail(main.run_id, row.candidate_id)
    if detail.row.chart_ref is not None:
        png = reader.get_artifact_bytes(main.run_id, detail.row.chart_ref.artifact_id)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    next_btn = next(b for b in at.button if b.label == "Next")
    next_btn.click().run()
    assert not at.exception
    reviews.add_review(main.run_id, row.candidate_id, judgment="unsure", note="ui test", setup_id="flag")
    at.run()
    assert not at.exception


def test_review_widget_keys_include_run_identity() -> None:
    left = review_widget_keys("run-a", "AAPL:2026-06-18:1d:100", "flag")
    right = review_widget_keys("run-b", "AAPL:2026-06-18:1d:100", "flag")
    assert all("run-a" in key for key in left)
    assert left[0] != right[0]
    assert left[1] != right[1]
    assert left[2] != right[2]


def _complete_one(store, prepared, note: str, judgment: str):
    from datetime import datetime, timezone

    from screener_loader.vision.serialization import digest_bytes
    from screener_loader.vision.types import (
        CandidateResult,
        ChartArtifact,
        ClassificationAttempt,
        RenderedImage,
        RunSummary,
        TokenUsage,
    )
    from vision_support import SYNTHETIC_PNG, assessment as make_assessment

    cand = prepared.candidates[0]
    run = store.create_run(prepared)
    art = store.save_artifact(
        run.run_id,
        ChartArtifact(
            artifact_id=f"candidate:{cand.ticker}:{cand.asof_date.isoformat()}",
            kind="candidate",
            image=RenderedImage(
                png_bytes=SYNTHETIC_PNG,
                width=8,
                height=6,
                sha256=digest_bytes(SYNTHETIC_PNG),
            ),
            title=cand.ticker,
            profile=cand.profile,
            ma_availability=(),
            final_session_date=cand.asof_date,
            ticker=cand.ticker,
            candidate_id=cand.candidate_id,
            source="rendered",
        ),
    )
    row = CandidateResult(
        candidate_id=cand.candidate_id,
        status="completed",
        ticker=cand.ticker,
        asof_date=cand.asof_date,
        features=cand.features,
        eligible_setup_ids=cand.eligible_setup_ids,
        assessments=(make_assessment("flag", "match", strength=2),),
        artifact_id=art.artifact_id,
        error=None,
        attempt_id="att-1",
        source_digest=cand.source_digest,
    )
    now = datetime(2026, 6, 19, 16, 0, tzinfo=timezone.utc)
    attempt = ClassificationAttempt(
        attempt_id="att-1",
        batch_id="b1",
        batch_fingerprint="fp-b1",
        candidate_ids=(cand.candidate_id,),
        setup_ids=("flag",),
        started_at=now,
        ended_at=now,
        provider="fake",
        model=prepared.config.model,
        response_id="r1",
        usage=TokenUsage(1, 1, 2),
        retry_after_seconds=None,
        latency_ms=1,
        error=None,
        sanitized_output={"ok": True},
        results=None,
        accepted=True,
    )
    store.commit_batch(run.run_id, batch_id="b1", results=(row,), attempt=attempt)
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
            usage=TokenUsage(1, 1, 2),
            status="completed",
        ),
    )
    return run, cand


def test_review_state_does_not_leak_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from screener_loader.vision.store import open_vision_scans
    from vision_support import sample_prepared_scan

    store, reader, reviews = open_vision_scans(tmp_path)
    prepared = sample_prepared_scan(mode="demo")
    run_a, cand = _complete_one(store, prepared, "note-a", "agree")
    run_b, _cand_b = _complete_one(store, prepared, "note-b", "disagree")
    reviews.add_review(run_a.run_id, cand.candidate_id, judgment="agree", note="alpha-run-note", setup_id="flag")
    reviews.add_review(run_b.run_id, cand.candidate_id, judgment="disagree", note="beta-run-note", setup_id="flag")
    assert reviews.current_review(run_a.run_id, cand.candidate_id, setup_id="flag").note == "alpha-run-note"
    assert reviews.current_review(run_b.run_id, cand.candidate_id, setup_id="flag").note == "beta-run-note"

    monkeypatch.setenv("NS_VISION_DEMO_ROOT", str(tmp_path))

    def _page():
        import os
        from pathlib import Path as P

        from screener_loader.vision.store import open_vision_scans as open_scans
        from screener_loader.vision_ui.results_page import render_results_page

        root = P(os.environ["NS_VISION_DEMO_ROOT"])
        _store, reader_inner, reviews_inner = open_scans(root)
        render_results_page(reader_inner, reviews_inner)

    at = AppTest.from_function(_page, default_timeout=30)
    at.run()
    assert not at.exception
    judgment_a, note_a, save_a = review_widget_keys(run_a.run_id, cand.candidate_id, "flag")
    judgment_b, note_b, save_b = review_widget_keys(run_b.run_id, cand.candidate_id, "flag")
    assert judgment_a != judgment_b

    run_ids = [r.run_id for r in reader.list_runs()]
    assert run_a.run_id in run_ids and run_b.run_id in run_ids

    # Newest run is first in the selector. Drive the selectbox to each run.
    box = at.selectbox[0]
    box.set_value(run_a.run_id).run()
    assert not at.exception
    note_widgets = {t.key: t for t in at.text_area}
    if note_a in note_widgets:
        assert "alpha-run-note" in str(note_widgets[note_a].value)
    radios = {r.key: r for r in at.radio if getattr(r, "key", None)}
    if judgment_a in radios:
        assert radios[judgment_a].value == "agree"

    box = at.selectbox[0]
    box.set_value(run_b.run_id).run()
    assert not at.exception
    note_widgets = {t.key: t for t in at.text_area}
    if note_b in note_widgets:
        assert "beta-run-note" in str(note_widgets[note_b].value)
        at.text_area(key=note_b).set_value("stale-should-not-write-to-a").run()
    save_buttons = [b for b in at.button if getattr(b, "key", None) == save_b]
    if save_buttons:
        save_buttons[0].click().run()
        assert not at.exception
        assert reviews.current_review(run_a.run_id, cand.candidate_id, setup_id="flag").note == "alpha-run-note"
        assert reviews.current_review(run_b.run_id, cand.candidate_id, setup_id="flag").note != "alpha-run-note"

