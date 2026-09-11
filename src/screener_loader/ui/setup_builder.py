"""Streamlit setup builder. Launch via `ns setups ui`."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import tempfile

import streamlit as st

from screener_loader.config import LoaderConfig
from screener_loader.paths import ensure_dirs
from screener_loader.setups.service import SetupService, update_spec_fields
from screener_loader.setups.spec import ChartStyle, SetupCriteria, SetupFilters, slugify_setup_id, VisionExample

_MILLION = 1_000_000.0
_CREATE_OPEN = "sb_show_create"
_ADD_EXAMPLE_OPEN = "sb_show_add_example"


def usd_to_millions(usd: float | None) -> float:
    if usd is None:
        return 0.0
    return float(usd) / _MILLION


def millions_to_usd(millions: float) -> float:
    return float(millions) * _MILLION


def adr_fraction_to_pct(fraction: float | None) -> float:
    if fraction is None:
        return 0.0
    return float(fraction) * 100.0


def adr_pct_to_fraction(pct: float) -> float:
    return float(pct) / 100.0


def format_millions_usd(usd: float | None) -> str:
    if usd is None:
        return "—"
    return f"${usd_to_millions(usd):g}M"


def _repo_root() -> Path:
    return Path(os.environ.get("NS_REPO_ROOT", ".")).resolve()


def _service() -> tuple[LoaderConfig, SetupService]:
    cfg = LoaderConfig(repo_root=_repo_root())
    ensure_dirs(cfg.paths)
    return cfg, SetupService(cfg.paths)


def _lines(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(text).splitlines() if s.strip())


def _join(items: tuple[str, ...]) -> str:
    return "\n".join(items)


def _init(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def render_builder_page() -> None:
    """Setup builder body. Page configuration belongs to the shared app entry."""
    cfg, svc = _service()
    specs = svc.list_setups()
    selected = st.session_state.get("setup_id")
    ids = [s.id for s in specs]
    if selected not in ids:
        selected = ids[0] if ids else None
        st.session_state["setup_id"] = selected

    col_list, col_main = st.columns([0.9, 3.1], gap="large")

    with col_list:
        head, plus = st.columns([3.2, 0.8])
        with head:
            st.markdown("**Setups**")
        with plus:
            if st.button("+", key="sb_plus_setup", help="New setup", width="stretch"):
                st.session_state[_CREATE_OPEN] = True
                st.rerun()

        if st.session_state.get(_CREATE_OPEN):
            new_name = st.text_input("Name", placeholder="Episodic Pivot", key="sb_new_name", label_visibility="visible")
            if new_name.strip():
                try:
                    st.caption(f"id: {slugify_setup_id(new_name)}")
                except Exception:
                    pass
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Create", type="primary", key="sb_create", width="stretch"):
                    try:
                        spec = svc.create(new_name.strip())
                        st.session_state["setup_id"] = spec.id
                        st.session_state[_CREATE_OPEN] = False
                        st.session_state.pop("sb_new_name", None)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with c2:
                if st.button("Cancel", key="sb_create_cancel", width="stretch"):
                    st.session_state[_CREATE_OPEN] = False
                    st.rerun()

        for spec in specs:
            is_sel = spec.id == selected
            label = spec.name if spec.enabled else f"{spec.name}  off"
            if is_sel:
                label = f"▸ {label}"
            if st.button(label, key=f"sel_{spec.id}", width="stretch", type="primary" if is_sel else "secondary"):
                st.session_state["setup_id"] = spec.id
                st.session_state[_ADD_EXAMPLE_OPEN] = False
                st.rerun()

    if not selected:
        with col_main:
            st.info("Create a setup to begin.")
        return

    spec = svc.get(selected)
    loaded = svc.load_examples(spec.id)
    by_id = {e.id: e for e in loaded}
    viewed = [by_id[eid] for eid in svc.compile_prompt(spec.id).example_ids if eid in by_id]

    with col_main:
        _render_config(svc, spec)
        _render_description(svc, spec)
        _render_examples(cfg, svc, spec, viewed)


def _render_config(svc: SetupService, spec) -> None:
    sid = spec.id
    g = svc.load_global_filters()
    _init(f"name_{sid}", spec.name)
    _init(f"en_{sid}", spec.enabled)
    _init(f"bars_{sid}", int(spec.lookback_bars))
    _init(f"vol_{sid}", bool(spec.chart.volume))
    _init(f"ma_{sid}", ",".join(str(x) for x in spec.chart.moving_averages))
    _init(f"use_px_{sid}", spec.filters.min_price is not None)
    _init(f"px_{sid}", float(spec.filters.min_price) if spec.filters.min_price is not None else 0.0)
    _init(f"use_dv_{sid}", spec.filters.min_dollar_vol_20d is not None)
    _init(
        f"dv_m_{sid}",
        usd_to_millions(spec.filters.min_dollar_vol_20d) if spec.filters.min_dollar_vol_20d is not None else 0.0,
    )
    _init(f"use_adr_{sid}", spec.filters.min_adr_pct_20 is not None)
    _init(
        f"adr_pct_{sid}",
        adr_fraction_to_pct(spec.filters.min_adr_pct_20) if spec.filters.min_adr_pct_20 is not None else 0.0,
    )

    top, save_col = st.columns([5.2, 0.8])
    with top:
        st.markdown("##### Config")
    with save_col:
        save_clicked = st.button("Save", type="primary", key=f"save_{sid}", width="stretch")

    r1c1, r1c2, r1c3 = st.columns([2.2, 1.0, 1.2])
    r1c1.text_input("Name", key=f"name_{sid}")
    r1c2.toggle("Enabled", key=f"en_{sid}")
    r1c3.selectbox("Timeframe", ["1d"], index=0, disabled=True, key=f"tf_{sid}")

    r2c1, r2c2, r2c3 = st.columns([1.0, 1.4, 2.6])
    r2c1.number_input("Bars", min_value=2, max_value=400, step=1, key=f"bars_{sid}")
    r2c2.checkbox("Volume on chart", key=f"vol_{sid}")
    r2c3.text_input("Moving averages", key=f"ma_{sid}")

    st.caption(
        f"Global floor: ${g.min_price:g} · {format_millions_usd(g.min_dollar_vol_20d)} 20d dollar vol. "
        "Setup filters can only tighten."
    )
    f1, f2, f3 = st.columns(3)
    with f1:
        st.checkbox("Min price", key=f"use_px_{sid}")
        st.number_input("USD", min_value=0.0, step=0.5, format="%.2f", key=f"px_{sid}")
    with f2:
        st.checkbox("Min 20d $ vol", key=f"use_dv_{sid}")
        st.number_input("Million USD", min_value=0.0, step=0.5, format="%.2f", key=f"dv_m_{sid}")
    with f3:
        st.checkbox("Min ADR", key=f"use_adr_{sid}")
        st.number_input("ADR %", min_value=0.0, step=0.1, format="%.1f", key=f"adr_pct_{sid}")
        st.caption("20-day mean of (H−L) / prior close.")

    st.caption("Market cap: unavailable")

    if save_clicked:
        _save_from_widgets(svc, spec)


def _render_description(svc: SetupService, spec) -> None:
    sid = spec.id
    _init(f"desc_{sid}", spec.description)
    _init(f"req_{sid}", _join(spec.criteria.required))
    _init(f"pref_{sid}", _join(spec.criteria.preferred))
    _init(f"dis_{sid}", _join(spec.criteria.disqualifiers))
    _init(f"notes_{sid}", spec.llm_notes)

    head, save_col = st.columns([5.2, 0.8])
    with head:
        st.markdown("##### Description")
    with save_col:
        if st.button("Save", type="primary", key=f"save_desc_{sid}", width="stretch"):
            _save_from_widgets(svc, spec)

    st.text_area("Overview", key=f"desc_{sid}", height=90)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_area("Required (one per line)", key=f"req_{sid}", height=140)
    with c2:
        st.text_area("Preferred (one per line)", key=f"pref_{sid}", height=140)
    with c3:
        st.text_area("Disqualifiers (one per line)", key=f"dis_{sid}", height=140)
    st.text_area("LLM notes", key=f"notes_{sid}", height=80)

    with st.expander("Compiled prompt"):
        st.code(svc.compile_prompt(spec.id).text)


def _save_from_widgets(svc: SetupService, spec) -> None:
    sid = spec.id
    _save_spec(
        svc,
        spec,
        name=st.session_state.get(f"name_{sid}", spec.name),
        enabled=bool(st.session_state.get(f"en_{sid}", spec.enabled)),
        timeframe=str(st.session_state.get(f"tf_{sid}", spec.timeframe)),
        lookback=int(st.session_state.get(f"bars_{sid}", spec.lookback_bars)),
        volume=bool(st.session_state.get(f"vol_{sid}", spec.chart.volume)),
        mas_text=str(st.session_state.get(f"ma_{sid}", ",".join(str(x) for x in spec.chart.moving_averages))),
        description=st.session_state.get(f"desc_{sid}", spec.description),
        required=st.session_state.get(f"req_{sid}", _join(spec.criteria.required)),
        preferred=st.session_state.get(f"pref_{sid}", _join(spec.criteria.preferred)),
        disqualifiers=st.session_state.get(f"dis_{sid}", _join(spec.criteria.disqualifiers)),
        llm_notes=st.session_state.get(f"notes_{sid}", spec.llm_notes),
        use_min_price=bool(st.session_state.get(f"use_px_{sid}", spec.filters.min_price is not None)),
        min_price=float(st.session_state.get(f"px_{sid}", spec.filters.min_price or 0.0)),
        use_min_dv=bool(st.session_state.get(f"use_dv_{sid}", spec.filters.min_dollar_vol_20d is not None)),
        min_dv_m=float(st.session_state.get(f"dv_m_{sid}", usd_to_millions(spec.filters.min_dollar_vol_20d))),
        use_min_adr=bool(st.session_state.get(f"use_adr_{sid}", spec.filters.min_adr_pct_20 is not None)),
        min_adr=float(st.session_state.get(f"adr_pct_{sid}", adr_fraction_to_pct(spec.filters.min_adr_pct_20))),
    )


def _save_spec(
    svc: SetupService,
    spec,
    *,
    name: str,
    enabled: bool,
    timeframe: str,
    lookback: int,
    volume: bool,
    mas_text: str,
    description: str,
    required: str,
    preferred: str,
    disqualifiers: str,
    llm_notes: str,
    use_min_price: bool,
    min_price: float,
    use_min_dv: bool,
    min_dv_m: float,
    use_min_adr: bool,
    min_adr: float,
) -> None:
    try:
        mas = tuple(int(x.strip()) for x in str(mas_text).split(",") if x.strip())
        updated = update_spec_fields(
            spec,
            name=name,
            enabled=enabled,
            lookback_bars=int(lookback),
            description=description,
            llm_notes=llm_notes,
            criteria=SetupCriteria(
                required=_lines(required),
                preferred=_lines(preferred),
                disqualifiers=_lines(disqualifiers),
            ),
            filters=SetupFilters(
                min_price=float(min_price) if use_min_price else None,
                min_dollar_vol_20d=millions_to_usd(min_dv_m) if use_min_dv else None,
                min_adr_pct_20=adr_pct_to_fraction(min_adr) if use_min_adr else None,
            ),
            chart=ChartStyle(volume=bool(volume), moving_averages=mas or spec.chart.moving_averages),
            timeframe=timeframe,
        )
        svc.save(updated)
        st.success("Saved")
        st.rerun()
    except Exception as e:
        st.error(str(e))


def _render_examples(cfg: LoaderConfig, svc: SetupService, spec, examples: list[VisionExample]) -> None:
    sid = spec.id
    idx_key = f"ex_idx_{sid}"
    _init(idx_key, 0)

    head, plus = st.columns([5.2, 0.8])
    with head:
        st.markdown("##### Examples")
    with plus:
        if st.button("+", key="sb_plus_example", help="Add example", width="stretch"):
            st.session_state[_ADD_EXAMPLE_OPEN] = True
            st.rerun()

    if examples:
        n = len(examples)
        idx = int(st.session_state.get(idx_key, 0)) % n
        st.session_state[idx_key] = idx
        nav1, nav2, nav3 = st.columns([0.7, 3.6, 0.7])
        with nav1:
            if st.button("‹", key=f"ex_prev_{sid}", width="stretch", disabled=n <= 1):
                st.session_state[idx_key] = (idx - 1) % n
                st.rerun()
        with nav2:
            ex = examples[idx]
            st.caption(_example_caption(ex, idx, n))
        with nav3:
            if st.button("›", key=f"ex_next_{sid}", width="stretch", disabled=n <= 1):
                st.session_state[idx_key] = (idx + 1) % n
                st.rerun()
        _render_one_example(cfg, spec, examples[st.session_state[idx_key] % n])
    else:
        st.caption("No examples yet.")

    if st.session_state.get(_ADD_EXAMPLE_OPEN):
        _render_add_example_form(cfg, svc, spec)


def _example_caption(ex: VisionExample, idx: int, n: int) -> str:
    loc = f"{ex.ticker} {ex.date}" if ex.type == "market_window" else str(ex.path)
    return f"{idx + 1} / {n}  ·  {ex.polarity}  ·  {ex.quality or 'unspecified'}  ·  {loc}"


def _render_one_example(cfg: LoaderConfig, spec, ex: VisionExample) -> None:
    if ex.note:
        st.caption(ex.note)
    try:
        from screener_loader.setups.charts import render_example_png

        st.image(render_example_png(cfg, spec, ex), width="stretch")
    except Exception as e:
        st.caption(str(e))


def _render_add_example_form(cfg: LoaderConfig, svc: SetupService, spec) -> None:
    st.markdown("**New example**")
    mode = st.radio("Source", ["Ticker/date", "Upload image"], horizontal=True, key="ex_mode")
    polarity = st.radio("Polarity", ["positive", "negative"], horizontal=True, key="ex_pol")
    quality = st.selectbox("Quality", ["canonical", "decent", "edge_case", "near_miss"], key="ex_qual")
    note = st.text_input("Notes", key="ex_note")

    if mode == "Ticker/date":
        ticker = st.text_input("Ticker", placeholder="NVDA", key="ex_ticker")
        asof = st.date_input("Date", value=date.today(), key="ex_date")
        if ticker:
            try:
                from screener_loader.setups.charts import load_ohlcv_window, render_chart_png

                df = load_ohlcv_window(cfg, ticker, asof, spec.lookback_bars)
                png = render_chart_png(df, ticker=ticker.upper(), asof_date=asof, style=spec.chart)
                st.image(png, caption=f"{ticker.upper()} {asof.isoformat()} (open-only last bar)")
            except Exception as e:
                st.caption(str(e))
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Add", type="primary", key="ex_add_window"):
                try:
                    svc.add_market_window_example(
                        spec.id,
                        ticker=ticker,
                        asof_date=asof,
                        polarity=polarity,
                        quality=quality,
                        note=note,
                    )
                    st.session_state[_ADD_EXAMPLE_OPEN] = False
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with a2:
            if st.button("Cancel", key="ex_add_cancel_w"):
                st.session_state[_ADD_EXAMPLE_OPEN] = False
                st.rerun()
        return

    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key="ex_upload")
    if uploaded is not None:
        st.image(uploaded.getvalue())
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Add", type="primary", key="ex_add_image"):
            if uploaded is None:
                st.error("Choose an image")
            else:
                suffix = Path(uploaded.name).suffix or ".png"
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)
                    svc.add_image_example(
                        spec.id,
                        tmp_path,
                        polarity=polarity,
                        quality=quality,
                        note=note,
                    )
                    st.session_state[_ADD_EXAMPLE_OPEN] = False
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)
    with a2:
        if st.button("Cancel", key="ex_add_cancel_i"):
            st.session_state[_ADD_EXAMPLE_OPEN] = False
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Setups", layout="wide")
    render_builder_page()


if __name__ == "__main__":
    main()
