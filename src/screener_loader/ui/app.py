"""Shared Streamlit entry point: setup catalog and vision screener."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import os
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PAGE_BUILDER = "builder"
PAGE_RESULTS = "results"
NAV_KEY = "ns_nav_page"
PAGE_SETUPS = "Setups"
PAGE_SCREENER = "Screener"
PAGE_LABELS = (PAGE_SETUPS, PAGE_SCREENER)

# Streamlit's default header (Deploy / toolbar) is position:fixed over the first
# row of the page. Hide it and keep enough top padding that Setups/Screener stay clickable.
# The page shell is full-bleed: the left rail stays at the window edge and the chart
# column stretches to the right edge.
_APP_CHROME = """
<style>
  [data-testid="stSidebar"] {display: none !important;}
  [data-testid="stSidebarCollapsedControl"] {display: none !important;}
  [data-testid="stHeader"] {display: none !important;}
  [data-testid="stToolbar"] {display: none !important;}
  [data-testid="stDecoration"] {display: none !important;}
  [data-testid="stStatusWidget"] {display: none !important;}
  .stAppToolbar, .stAppDeployButton, header[data-testid="stHeader"] {display: none !important;}
  .stApp > header {display: none !important;}
  [data-testid="stMainBlockContainer"],
  .block-container,
  section.main > div.block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0.65rem !important;
    padding-right: 0.65rem !important;
  }
  .st-key-ns_page {
    width: 100% !important;
    max-width: 100% !important;
  }
  .st-key-ns_chart {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
    max-width: none !important;
  }
  .st-key-ns_chart [data-testid="stFullScreenFrame"] > div,
  .st-key-ns_chart [data-testid="stImage"],
  .st-key-ns_chart [data-testid="stImageContainer"],
  .st-key-ns_chart [data-testid="stImageContainer"] img {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
  }
</style>
"""


def _repo_root() -> Path:
    return Path(os.environ.get("NS_REPO_ROOT", ".")).resolve()


def _initial_page() -> str:
    raw = os.environ.get("NS_VISION_PAGE", PAGE_BUILDER).strip().lower()
    if raw in {PAGE_RESULTS, "results", "view", "screener"}:
        return PAGE_RESULTS
    return PAGE_BUILDER


def _render_header(st) -> None:
    st.markdown(_APP_CHROME, unsafe_allow_html=True)
    st.segmented_control(
        "Page",
        PAGE_LABELS,
        key=NAV_KEY,
        label_visibility="collapsed",
        width="content",
        required=True,
    )
    st.divider()


def main() -> None:
    import streamlit as st

    from screener_loader.dotenv import load_dotenv

    load_dotenv(_repo_root() / ".env", override=False)

    st.set_page_config(page_title="Nuanced Screener", layout="wide", initial_sidebar_state="collapsed")
    query_page = str(st.query_params.get("page", "") or "").strip().lower()
    default = PAGE_RESULTS if query_page in {PAGE_RESULTS, "results", "view", "screener"} else _initial_page()
    if NAV_KEY not in st.session_state:
        st.session_state[NAV_KEY] = PAGE_SCREENER if default == PAGE_RESULTS else PAGE_SETUPS

    run_hint = os.environ.get("NS_VISION_RUN_ID") or str(st.query_params.get("run", "") or "")
    if run_hint:
        from screener_loader.vision_ui.components import state_key

        st.session_state.setdefault(state_key("run_id"), run_hint)

    _render_header(st)

    if st.session_state[NAV_KEY] == PAGE_SETUPS:
        from screener_loader.ui.setup_builder import render_builder_page

        render_builder_page()
        return

    from screener_loader.config import LoaderConfig
    from screener_loader.ui.scan_jobs import FOLLOW_JOB_KEY, LIVE_REFRESH_SECONDS, should_live_refresh
    from screener_loader.ui.scan_panel import render_scan_panel
    from screener_loader.vision.service import default_scan_root
    from screener_loader.vision.store import open_vision_scans
    from screener_loader.vision_ui.components import state_key
    from screener_loader.vision_ui.results_page import render_results_detail, render_results_list

    cfg = LoaderConfig(repo_root=_repo_root())
    page = st.container(horizontal=True, wrap=False, gap="small", vertical_alignment="top", key="ns_page")
    col_side = page.container(width=320)
    col_main = page.container(width="stretch", key="ns_chart")

    with col_side:
        job = render_scan_panel(cfg)

        polling = should_live_refresh(job)

        def _sidebar_results() -> None:
            from screener_loader.ui.scan_panel import render_scan_status

            _store, reader, _reviews = open_vision_scans(default_scan_root(cfg))
            render_scan_status(job, reader)
            follow = bool(st.session_state.get(FOLLOW_JOB_KEY))
            override = job.run_id if follow and job is not None and job.run_id else None
            render_results_list(reader, include_run_picker=False, run_id_override=override)
            if polling and job is not None and not job.is_active():
                st.rerun(scope="app")

        if polling:
            st.fragment(_sidebar_results, run_every=timedelta(seconds=LIVE_REFRESH_SECONDS))()
        else:
            _sidebar_results()

    with col_main:
        def _main_detail() -> None:
            _store, reader, reviews = open_vision_scans(default_scan_root(cfg))
            render_results_detail(reader, reviews)

        if should_live_refresh(job):
            st.fragment(_main_detail, run_every=timedelta(seconds=LIVE_REFRESH_SECONDS))()
        else:
            _main_detail()


if __name__ == "__main__":
    main()
