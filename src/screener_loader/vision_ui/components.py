"""Page-local helpers for the vision results UI. No Streamlit page config."""

from __future__ import annotations

from typing import Sequence

from screener_loader.vision.query import format_adr_display_pct
from screener_loader.vision.types import (
    CandidateRow,
    CandidateStatus,
    FeatureValue,
    PageRequest,
    ResultQuery,
    SortSpec,
    Verdict,
)

STATE_PREFIX = "vr_"
DEFAULT_PAGE_SIZE = 10
UI_LIST_PAGE_SIZE = 10_000

VIEW_MATCHES = "matches"
VIEW_ALL = "all"
VIEW_NO_MATCHES = "no_matches"
VIEW_UNCERTAIN = "uncertain"
VIEW_ERRORS = "errors"

VIEW_LABELS = {
    VIEW_MATCHES: "Matches",
    VIEW_ALL: "All candidates",
    VIEW_NO_MATCHES: "No matches",
    VIEW_UNCERTAIN: "Uncertain",
    VIEW_ERRORS: "Errors / skipped",
}

JUDGMENT_LABELS = {
    "agree": "Agree with model",
    "disagree": "Disagree (reject flag or record missed setup)",
    "unsure": "Unsure",
}


def state_key(name: str) -> str:
    """Namespace widget/session keys so they never collide with the builder's ``setup_id``."""

    if name == "setup_id":
        raise ValueError("results UI must not use the builder setup_id key")
    return STATE_PREFIX + name


def review_widget_keys(run_id: str, candidate_id: str, setup_id: str) -> tuple[str, str, str]:
    """Run-scoped review widget keys so switching runs cannot leak judgments/notes."""

    suffix = f"{run_id}_{candidate_id}_{setup_id}"
    return (
        state_key(f"judgment_{suffix}"),
        state_key(f"note_{suffix}"),
        state_key(f"save_{suffix}"),
    )


def reconcile_selected_candidate(selected_id: str | None, ordered_ids: Sequence[str]) -> str | None:
    """Keep the selected candidate when it remains in the filtered sequence; otherwise take the first.

    Predictable rule used by the page and tests:
    1. If ``selected_id`` is still in ``ordered_ids``, keep it (even if the page would change).
    2. If the filtered sequence is non-empty, select the first id.
    3. If the sequence is empty, clear the selection (``None``).
    Switching runs is handled by the caller passing the new run's ids (the old id will miss).
    """

    ids = list(ordered_ids)
    if selected_id and selected_id in ids:
        return selected_id
    if ids:
        return ids[0]
    return None


def view_query_parts(view: str) -> tuple[tuple[Verdict, ...], tuple[CandidateStatus, ...]]:
    if view == VIEW_MATCHES:
        return ("match",), ()
    if view == VIEW_NO_MATCHES:
        return ("no_match",), ()
    if view == VIEW_UNCERTAIN:
        return ("uncertain",), ()
    if view == VIEW_ERRORS:
        return (), ("error", "skipped")
    return (), ()


def build_result_query(
    *,
    run_id: str,
    view: str = VIEW_MATCHES,
    setup_ids: Sequence[str] = (),
    min_match_strength: int | None = None,
    max_match_strength: int | None = None,
    review_state: str = "any",
    sort_key: str = "match_strength",
    descending: bool = True,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[ResultQuery, tuple[CandidateStatus, ...]]:
    verdicts, statuses = view_query_parts(view)
    query = ResultQuery(
        run_id=run_id,
        setup_ids=tuple(setup_ids),
        verdicts=verdicts,
        min_match_strength=min_match_strength,
        max_match_strength=max_match_strength,
        review_state=review_state,  # type: ignore[arg-type]
        sort=SortSpec(key=sort_key, descending=descending),  # type: ignore[arg-type]
        page=PageRequest(page=max(1, page), page_size=page_size),
    )
    return query, statuses


def format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def format_dollar_volume(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}/day"


def format_dollar_vol_millions(value: float | None) -> str:
    if value is None:
        return "—"
    millions = float(value) / 1_000_000.0
    return f"${millions:,.1f}M"


def format_adr_pct(fraction: float | None) -> str:
    pct = format_adr_display_pct(fraction)
    if pct is None:
        return "—"
    return f"{pct:.1f}%"


def md_escape_dollars(text: str) -> str:
    """Streamlit captions parse `$...$` as LaTeX; escape currency for markdown."""

    return text.replace("$", "\\$")


def feature_caption(features: FeatureValue) -> str:
    units = features.units()
    return (
        f"{format_price(features.close)} ({units['close']}) · "
        f"{format_dollar_volume(features.dollar_vol_avg_20)} ({units['dollar_vol_avg_20']}) · "
        f"ADR {format_adr_pct(features.adr_pct_20)} (stored as {units['adr_pct_20']})"
    )


def badge_text(row: CandidateRow) -> str:
    if not row.matched_setups:
        if row.status != "completed":
            return row.status
        return "no match"
    parts = [f"{b.setup_name} {b.match_strength}" for b in row.matched_setups]
    return " · ".join(parts)


def run_status_label(status: str, *, synthetic: bool, mode: str) -> str:
    bits = [status]
    if mode and mode != "live":
        bits.append(mode)
    if synthetic:
        bits.append("synthetic demo")
    return " · ".join(bits)


def unavailable_text(value: int | None, *, unavailable: Sequence[str], name: str) -> str:
    if value is None or name in unavailable:
        return "unavailable"
    return str(value)
