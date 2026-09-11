"""Mountable Streamlit results page. Does not set page config or launch scans."""

from __future__ import annotations

from typing import Any

from screener_loader.vision.query import FilesystemResultReader, NeighborPosition
from screener_loader.vision.reviews import FilesystemReviewStore
from screener_loader.vision.types import ResultQuery, SetupSnapshot, VisionError

from .components import (
    UI_LIST_PAGE_SIZE,
    VIEW_ALL,
    VIEW_LABELS,
    build_result_query,
    format_adr_pct,
    format_dollar_vol_millions,
    md_escape_dollars,
    reconcile_selected_candidate,
    review_widget_keys,
    run_status_label,
    state_key,
    unavailable_text,
)

_STREAMLIT = None
_SNAPSHOT_KEY = "snapshot"


def _st():
    global _STREAMLIT
    if _STREAMLIT is None:
        import streamlit as st

        _STREAMLIT = st
    return _STREAMLIT


def _clear_snapshot(st: Any) -> None:
    st.session_state[state_key(_SNAPSHOT_KEY)] = None


def _store_snapshot(
    st: Any,
    *,
    run_id: str,
    selected_id: str | None,
    query: ResultQuery,
    ticker_query: str,
    statuses: tuple,
) -> None:
    st.session_state[state_key(_SNAPSHOT_KEY)] = {
        "run_id": run_id,
        "selected_id": selected_id,
        "query": query,
        "ticker_query": ticker_query,
        "statuses": statuses,
    }


def render_results_page(
    reader: FilesystemResultReader,
    reviews: FilesystemReviewStore,
    *,
    page_size: int = UI_LIST_PAGE_SIZE,
) -> None:
    """List/detail viewer over persisted runs. Refresh reads the store only."""

    st = _st()
    list_col, detail_col = st.columns([0.9, 3.1], gap="large")
    with list_col:
        render_results_list(reader, page_size=page_size)
    with detail_col:
        render_results_detail(reader, reviews)


_LIST_CSS = """
<style>
  .st-key-ns_result_list [data-testid="stVerticalBlock"] {
    gap: 0 !important;
  }
  .st-key-ns_result_list div[data-testid="stButton"] {
    margin: 0 !important;
  }
  .st-key-ns_result_list div[data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(128,128,128,0.28) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    min-height: 1.45rem !important;
    padding: 0.12rem 0.4rem !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
  }
  .st-key-ns_result_list div[data-testid="stButton"] button[kind="primary"] {
    background: rgba(128,128,128,0.18) !important;
  }
  .st-key-ns_keynav {
    position: absolute !important;
    left: -9999px !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
  }
  .st-key-vr_kb_prev, .st-key-vr_kb_next {
    position: absolute !important;
    left: -9999px !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
  }
</style>
"""

_KEYNAV_HTML = """
<script>
(function() {
  const w = window.parent;
  const doc = w.document;
  const handler = function(e) {
    if (e.repeat) return;
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.target && e.target.isContentEditable) return;
    let label = null;
    if (e.key === "ArrowDown" || e.key === " " || e.code === "Space") label = "Next";
    else if (e.key === "ArrowUp") label = "Previous";
    else return;
    e.preventDefault();
    const buttons = Array.from(doc.querySelectorAll("button"));
    const btn = buttons.find(b => (b.textContent || "").trim() === label && !b.disabled);
    if (btn) btn.click();
  };
  if (w.__nsKeyNav) doc.removeEventListener("keydown", w.__nsKeyNav);
  w.__nsKeyNav = handler;
  doc.addEventListener("keydown", handler);
})();
</script>
"""


def render_results_list(
    reader: FilesystemResultReader,
    *,
    page_size: int = UI_LIST_PAGE_SIZE,
    include_run_picker: bool = True,
    run_id_override: str | None = None,
) -> None:
    """Filters and compact ticker rows for the screener sidebar."""

    st = _st()
    st.markdown(_LIST_CSS, unsafe_allow_html=True)
    runs = reader.list_runs()
    if not runs:
        _clear_snapshot(st)
        st.caption("No results yet.")
        return

    run_labels = {
        run.run_id: (
            f"{run.created_at.strftime('%Y-%m-%d %H:%M')} · {run.status}"
            f"{' · demo' if run.synthetic else ''} · {run.run_id[-8:]}"
        )
        for run in runs
    }
    run_ids = [r.run_id for r in runs]
    if state_key("view") not in st.session_state:
        st.session_state[state_key("view")] = VIEW_ALL

    if include_run_picker:
        if state_key("run_id") not in st.session_state or st.session_state[state_key("run_id")] not in run_ids:
            st.session_state[state_key("run_id")] = run_ids[0]
        with st.expander("Debug"):
            st.selectbox(
                "Run",
                options=run_ids,
                format_func=lambda rid: run_labels.get(rid, rid),
                key=state_key("run_id"),
            )
            if st.button("Refresh", key=state_key("refresh"), width="stretch"):
                st.rerun()
        run_id = str(st.session_state[state_key("run_id")])
    elif run_id_override:
        run_id = str(run_id_override)
    else:
        run_id = str(st.session_state.get(state_key("run_id")) or run_ids[0])
        if run_id not in run_ids:
            run_id = run_ids[0]

    prev_run = st.session_state.get(state_key("prev_run_id"))
    if prev_run is not None and prev_run != run_id:
        st.session_state[state_key("candidate_id")] = None
        st.session_state[state_key("page")] = 1
    st.session_state[state_key("prev_run_id")] = run_id

    try:
        stored = reader.load_run(run_id)
        frozen = reader.load_frozen_inputs(run_id)
    except VisionError as exc:
        _clear_snapshot(st)
        st.error(str(exc))
        return

    counts = reader.extended_counts(run_id)
    setups = list(frozen.setups)
    setup_options = [s.setup_id for s in setups]
    st.caption(
        f"{run_status_label(stored.status, synthetic=stored.synthetic, mode=stored.config.mode)} · "
        f"{counts.completed}/{counts.candidates} settled · {counts.setup_matches} matches"
    )
    with st.expander("Filters"):
        view = st.selectbox(
            "View",
            options=list(VIEW_LABELS.keys()),
            format_func=lambda key: VIEW_LABELS[key],
            key=state_key("view"),
        )
        ticker_query = st.text_input("Ticker search", key=state_key("ticker"))
        selected_setups = st.multiselect(
            "Setups (ANY)",
            options=setup_options,
            format_func=lambda sid: next((s.name for s in setups if s.setup_id == sid), sid),
            key=state_key("setup_ids"),
        )
        strength = st.selectbox(
            "Min strength",
            options=("any", 1, 2, 3),
            key=state_key("min_strength"),
        )
        review_state = st.selectbox(
            "Review",
            options=("any", "unreviewed", "agree", "disagree", "unsure"),
            key=state_key("review_state"),
        )
        asof_dates = sorted({c.asof_date for c in frozen.candidates})
        asof_label = ", ".join(d.isoformat() for d in asof_dates) if asof_dates else "n/a"
        diag = frozen.diagnostics
        st.caption(
            f"Chart session {asof_label} · lookback {frozen.profile.lookback_bars} bars"
        )
        st.caption(
            "Eligibility: "
            f"universe {unavailable_text(diag.universe_count, unavailable=diag.unavailable, name='universe_count')} · "
            f"input {unavailable_text(diag.input_count, unavailable=diag.unavailable, name='input_count')} · "
            f"eligible union {unavailable_text(diag.eligible_union_count, unavailable=diag.unavailable, name='eligible_union_count')} · "
            f"not eligible {unavailable_text(diag.not_eligible_count, unavailable=diag.unavailable, name='not_eligible_count')}"
        )

    view = str(st.session_state.get(state_key("view")) or VIEW_ALL)
    ticker_query = str(st.session_state.get(state_key("ticker")) or "")
    selected_setups = list(st.session_state.get(state_key("setup_ids")) or [])
    strength = st.session_state.get(state_key("min_strength"), "any")
    review_state = str(st.session_state.get(state_key("review_state") or "any"))
    min_strength = None if strength == "any" else int(strength)
    query, statuses = build_result_query(
        run_id=run_id,
        view=view,
        setup_ids=selected_setups,
        min_match_strength=min_strength,
        review_state=review_state,
        sort_key="arrival",
        descending=False,
        page=1,
        page_size=page_size,
    )
    if view == VIEW_ALL and not statuses:
        statuses = ("completed", "error")
    ordered_ids = reader.ordered_ids(query, ticker_query=ticker_query, statuses=statuses)
    selected_id = reconcile_selected_candidate(
        st.session_state.get(state_key("candidate_id")),
        ordered_ids,
    )
    st.session_state[state_key("candidate_id")] = selected_id
    _store_snapshot(
        st,
        run_id=run_id,
        selected_id=selected_id,
        query=query,
        ticker_query=ticker_query,
        statuses=statuses,
    )
    result_page = reader.query(query, ticker_query=ticker_query, statuses=statuses)
    pos = (
        reader.neighbor(query, selected_id, ticker_query=ticker_query, statuses=statuses)
        if selected_id
        else None
    )
    _render_keyboard_nav(st, pos)

    if not result_page.items:
        st.caption("No results match these filters.")
        return
    with st.container(height=720, border=False, key="ns_result_list"):
        for row in result_page.items:
            selected = row.candidate_id == selected_id
            if st.button(
                row.ticker,
                key=state_key(f"pick_{run_id}_{row.candidate_id}"),
                type="primary" if selected else "secondary",
                width="stretch",
            ):
                st.session_state[state_key("candidate_id")] = row.candidate_id
                st.rerun()
    if pos is not None:
        st.caption(f"{pos.position} of {pos.total}")


def render_results_detail(
    reader: FilesystemResultReader,
    reviews: FilesystemReviewStore,
) -> None:
    """Chart and assessments for the candidate selected in the sidebar."""

    st = _st()
    snap = st.session_state.get(state_key(_SNAPSHOT_KEY))
    if not snap or not snap.get("run_id") or not snap.get("selected_id"):
        return

    run_id = str(snap["run_id"])
    selected_id = str(snap["selected_id"])
    try:
        frozen = reader.load_frozen_inputs(run_id)
        detail = reader.get_detail(run_id, selected_id)
    except VisionError as exc:
        st.error(str(exc))
        return
    row = detail.row
    st.markdown(
        md_escape_dollars(
            f"**{row.ticker}** · {row.asof_date.isoformat()} · "
            f"ADR {format_adr_pct(row.features.adr_pct_20)} · {format_dollar_vol_millions(row.features.dollar_vol_avg_20)}"
        )
    )
    _render_chart(st, reader, run_id, row)
    snapshots = {s.setup_id: s for s in frozen.setups}
    current_reviews = {r.setup_id: r for r in reviews.current_reviews_for_candidate(run_id, selected_id)}
    for assessment in detail.assessments:
        snap_setup = snapshots.get(assessment.setup_id)
        _render_assessment(st, assessment, snap_setup, current_reviews.get(assessment.setup_id), reviews, run_id, selected_id)


def _render_keyboard_nav(st: Any, pos: NeighborPosition | None) -> None:
    """Hidden Previous/Next targets for arrow/space keyboard iteration."""

    import streamlit.components.v1 as components

    prev_disabled = pos is None or pos.prev_id is None
    next_disabled = pos is None or pos.next_id is None
    with st.container(key="ns_keynav"):
        cols = st.columns(2)
        with cols[0]:
            if st.button("Previous", disabled=prev_disabled, key=state_key("kb_prev")):
                if pos is not None and pos.prev_id:
                    st.session_state[state_key("candidate_id")] = pos.prev_id
                    st.rerun()
        with cols[1]:
            if st.button("Next", disabled=next_disabled, key=state_key("kb_next")):
                if pos is not None and pos.next_id:
                    st.session_state[state_key("candidate_id")] = pos.next_id
                    st.rerun()
    components.html(_KEYNAV_HTML, height=0)


def _render_chart(st: Any, reader: FilesystemResultReader, run_id: str, row) -> None:
    if row.chart_ref is None:
        return
    try:
        png = reader.get_artifact_bytes(run_id, row.chart_ref.artifact_id)
    except VisionError:
        return
    st.image(png, width="stretch")


def _render_assessment(
    st: Any,
    assessment,
    snap: SetupSnapshot | None,
    current_review,
    reviews: FilesystemReviewStore,
    run_id: str,
    candidate_id: str,
) -> None:
    name = snap.name if snap is not None else assessment.setup_id
    st.markdown(f"**{name}**")
    if assessment.reason:
        st.write(assessment.reason)
    judgment_key, note_key, save_key = review_widget_keys(run_id, candidate_id, assessment.setup_id)
    default_judgment = current_review.judgment if current_review is not None else "unsure"
    options = ("agree", "disagree", "unsure")
    st.radio(
        "Review",
        options=options,
        format_func=lambda j: {"agree": "Agree", "disagree": "Disagree", "unsure": "Unsure"}[j],
        index=options.index(default_judgment) if default_judgment in options else 2,
        key=judgment_key,
        horizontal=True,
    )
    st.text_area("Notes", value=current_review.note if current_review else "", key=note_key, height=70)
    if st.button("Save", key=save_key):
        reviews.add_review(
            run_id,
            candidate_id,
            judgment=st.session_state[judgment_key],
            note=st.session_state.get(note_key) or "",
            setup_id=assessment.setup_id,
        )
        st.rerun()
