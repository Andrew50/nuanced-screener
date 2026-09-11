# Vision scan MVP

Chart-based setup classification over the existing YAML catalog, last-N bars, and ticker universe. This is an offline-first local pipeline. It does not train models, backtest, schedule jobs, or host a public frontend.

Contracts: [VISION_CONTRACT.md](VISION_CONTRACT.md). Agent handoffs: [vision_handoffs/pipeline.md](vision_handoffs/pipeline.md), [vision_handoffs/results.md](vision_handoffs/results.md), [vision_handoffs/integration.md](vision_handoffs/integration.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui,vision]"
cp .env.example .env
```

- Base install (`pip install -e ".[dev]"`) keeps `ns setups`, `ns query`, `ns models`, and `ns candidates` working without OpenAI, Streamlit, or matplotlib.
- `ui` extra: Streamlit + matplotlib (builder, results, `ns setups ui`, `ns vision view`).
- `vision` extra: OpenAI SDK + matplotlib (headless `ns screen` / `ns vision scan` without Streamlit).
- Installed wheels include `screener_loader/vision/response_schema.json`.

`.env` placeholders only: `OPENAI_API_KEY`, `NS_VISION_MODEL`. A missing key refuses live scans. It never substitutes demo/fake predictions.

## Prerequisites (existing setup/data path)

1. Catalog via `ns setups` / `ns setups ui` (`SetupService` YAML under `data/setups/`). Enable the setups you want. Criteria have no persistent IDs; scans snapshot `{setup_id}:required:{index}` (and preferred/disqualifier) at freeze time.
2. Universe: `data/meta/tickers.csv` (`ns universe`). The literal ticker `NA` is preserved.
3. OHLCV + derived last-N: `ns update` then `ns rebuild-last100 --window-size N` writing `data/derived/last_100_bars.parquet`. The pipeline will not rebuild this file.
4. Example charts: uploaded png/jpg/webp bytes, or market windows loaded with `load_ohlcv_window` (small catalog only). Missing example history fails before API calls; it is not backfilled.

Market-cap filters remain unavailable and fail the whole operation (`MarketCapUnavailableError`). Gap/premarket/ranked feeds are out of scope.

## Compatible chart profile

One profile per run. Enabled setups must agree on `timeframe`, `lookback_bars`, `chart.volume`, and ordered `chart.moving_averages`. Conflicts are named (field + per-setup values) and abort before provider calls. Saved YAML values win; 100 bars and SMA 10/20/50 are new-setup defaults, not overrides. Lookbacks 2–400 are allowed.

SMAs use **displayed bars only** (`min_periods=period`). There is no pre-display warm-up. Periods longer than the window stay configured and are reported unavailable.

## Last-N, freshness, diagnostics

- Eligible candidates are the eligibility union intersected with `tickers.csv`. No `candidates.py` prefilter.
- Bulk windows come from `last_100_bars.parquet`. Larger stored N is sliced to the profile lookback. If the file’s max `rn` is below lookback, fail with `ns rebuild-last100 --window-size N --repo-root <root>`. Isolated short or stale tickers are skipped, not silently shortened.
- Freshness is latest-daily only: last-N must not be newer than the last closed NYSE session, and for live scans must not be older than the session delayed grouped-daily vendors authorize (previous session while “today” is still the calendar date). `EligibilityResult.asof_date` is a maximum, not proof. Globally stale or future snapshots fail. Isolated stale inputs skip. Reading saved results does not re-check freshness. Live `ns screen` auto-runs `ns update` when the update stamp is older than 24 hours.
- Features: `close` (USD/share), `dollar_vol_avg_20` (USD/day), `adr_pct_20` (fraction; 0.04 = 4%). Pandas missing values become JSON null. Filter-rejection reasons are unavailable (`not_eligible` is membership, not a ledger).
- Polygon OHLC are already adjusted at source. The adapter does not apply `adj_close` or adjust again. Unknown provenance is `"unknown"`; current config/mtime is not historical proof.

## Commands

Scan root: `data/vision_scans/` (gitignored).

```bash
# Dry-run: render, compile, estimate. Zero provider calls.
ns screen --dry-run --repo-root .

# Labeled local classifier (explicit; not a missing-key fallback)
ns screen --demo --max-candidates 5 --repo-root .

# Live (requires OPENAI_API_KEY and an explicit model id)
ns screen --model "$NS_VISION_MODEL" --max-candidates 3 --repo-root .

ns vision resume --run-id run-... --repo-root .
ns vision runs --repo-root .
ns vision export --run-id run-... --out-dir /tmp/vision-export --repo-root .
ns vision evaluate --run-id run-... --repo-root .
ns vision view --run run-... --port 8501 --repo-root .
ns setups ui --port 8501 --repo-root .    # same app, default builder page
```

`--max-candidates` is a smoke-test cap after eligibility. Do not use it as a production sampling policy. A live market-wide scan is not launched by default.

## UI

`ns setups ui` and `ns vision view` share `screener_loader/ui/app.py`. Page config is set once. Sidebar switches **Setup builder** and **Results**.

- Setup builder: YAML catalog, filters, examples. `python -m streamlit run src/screener_loader/ui/setup_builder.py` still works.
- Results: browse persisted runs. **Run a scan** starts a background job; a progress strip and the candidate list refresh as each batch is committed (no implicit new scan on Refresh). Live still needs `OPENAI_API_KEY` and a model id; a missing key does not invent results. Uncapped / market-wide live scans stay on `ns screen` (`ns vision scan` is an alias).

## Reviews and evaluation

Human reviews are append-only and separate from model verdicts and execution errors. Offline `ns vision evaluate` uses explicit reviewed candidate/setup pairs (`agree`/`disagree`). Unsure and unreviewed are excluded. Prompt examples are not held-out cases. Precision/recall are scoped per setup; undefined denominators stay null. A rejected setup is not a negative for every setup. Reviewing only match flags cannot establish whole-universe recall.

## Deferred

Alerts, scheduling, streaming, backtesting, model training, automatic example promotion, hosted frontends, market-cap coverage, gap/premarket/ranked feeds, mixed-vendor provenance reconstruction, default-branch merge, paid market-wide scans.
