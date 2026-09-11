# Vision scan contracts

Frozen interfaces for the chart → request → classification → resumable scan pipeline. Agent 1 owns this document and the core models. Agents 2 and 3 consume it; they must not fork types.

Python 3.10. Core modules (`types`, `protocols`, `serialization`) import neither OpenAI, Streamlit, nor matplotlib. Snapshot helpers may import existing setup models and `compile_prompt` for example ordering only.

## Ownership

| Area | Owner | Paths |
| --- | --- | --- |
| Contracts, snapshots, schema, compiler, classifier attempt, batching, runner | Agent 1 | `src/screener_loader/vision/{types,protocols,serialization,snapshots,prompts,validation,client,batching,scan,charts,response_schema.json}.py`, `setups/charts.py` (narrow renderer hardening) |
| Run persistence, queries, reviews, results UI | Agent 2 | storage + Streamlit results page |
| Real `ScanSource`, CLI/deps, app navigation, integration | Agent 3 | adapters, `pyproject.toml`, root README, `.env.example`, CLI, shared Streamlit shell |

Do not edit schema/catalog/service/eligibility, feature SQL, data sync, global paths, or legacy ML. The chart renderer is Agent 1's only allowed exception in protected plotting code. New vision snapshots are immutable run inputs, not another editable catalog.

YAML builder entities stay as they are: `SetupService`, `SetupSpec`, `SetupCriteria`, `VisionExample`, `ChartStyle`, `SetupFilters`, `compute_eligibility`, `ui/setup_builder.py`. Only snapshot helpers and Agent 3 adapters bind those entities to vision contracts.

## Bindings to existing setup code

- Snapshot a `SetupSpec` plus the original `setup.yaml` bytes (`yaml_digest`) and a canonical content digest (`content_digest`). Preserve global-filter YAML bytes and example source content (upload bytes or window source digest).
- Criteria have no persistent IDs. At snapshot time assign `{setup_id}:required:{zero_based_index}`, `{setup_id}:preferred:{zero_based_index}`, `{setup_id}:disqualifier:{zero_based_index}`. Those IDs mean something only for that snapshot digest. No slugification and no catalog migration.
- Use **all** examples for each relevant setup. Ordering is `compile_prompt(spec, examples).example_ids`: polarity (positive then negative), quality (`canonical`, `decent`, `edge_case`, `near_miss`, unspecified), then example id. `setups/prompt.py` remains the builder text preview. The multimodal compiler consumes structured snapshots and images.
- Example ids are unique per setup, not globally. Scoped identity is `{setup_id}/{example_id}` (`snapshots.scoped_example_id`). The same catalog example id may appear in two setups.
- One compatible `ChartProfile` per run: compare `timeframe`, `lookback_bars`, `chart.volume`, and ordered `chart.moving_averages` across enabled setups. Fail with `ChartProfileConflictError` listing each conflicting field and per-setup value before any API call. Saved settings win; 100 bars and SMA 10/20/50 are defaults for new setups, not overrides. The builder UI allows lookbacks 2–400; the profile records the saved lookback as-is (minimum 2).
- `setups.charts.render_chart_figure` / `render_chart_png` remain the sole plotting implementation. `vision/charts.py` is a thin wrapper. Builder imports stay direct; there is no renderer injection hook.
- SMAs use **displayed bars only** with `min_periods=period`. No pre-display warm-up. Keep configured periods even when the window is too short to produce values, and report unavailability (`moving_average_availability`).
- `ScanSource` (Agent 3) binds `SetupService` + `compute_eligibility` + bulk last-N + `load_universe(config)` / `tickers.csv`. Intersect eligibility with the configured universe. Do not use `candidates.py`, chart-shape prefilters, supervised labels, or open-only window masking.
- Available features: `close` (USD/share), `dollar_vol_avg_20` (USD/day), `adr_pct_20` (fraction; 0.04 = 4%). Normalize pandas missing values to JSON null. Market-cap filters stay unavailable and raise the existing `MarketCapUnavailableError` for the whole operation.
- Eligibility is membership, not a rejection ledger. Record only independently measured universe/input/union/per-setup eligible counts. Do not invent `not_eligible` as `universe - eligible`. Missing membership is `not_eligible` with unknown reason. Unavailable diagnostics are `null` / `ScanDiagnostics.unavailable`, never guessed counts. Do not duplicate `_setup_mask` for diagnostics.
- Bulk bars come from `DataPaths.last_100_bars_parquet`. Inspect capacity and slice to the agreed lookback. If the derived file is globally too short, raise `InsufficientLastNError` naming `ns rebuild-last100 --window-size N`. Do not rebuild implicitly. Isolated short windows are `InputSkip(kind="short_window")`. Historical example charts may use `load_ohlcv_window`; never call it per stock for a market-wide scan.
- Freshness (Agent 3): latest daily snapshot only. Capture current UTC time, use calendar close times for the latest completed session, and treat delayed grouped-daily authorization (never same-calendar-day) as the live floor. Last-N may sit on that authorized previous session after the cash close. `EligibilityResult.asof_date` is a maximum, not proof of freshness. Fail a globally stale or future snapshot (`StaleSnapshotError`). Record isolated stale candidates as skipped. No arbitrary historical screening cutoff. Polygon OHLC are already adjusted by their source; do not substitute `adj_close` or adjust again. Unknown provenance is the string `"unknown"`.
- A stock may match zero, one, or several setups. Persist every successfully rendered candidate chart (including non-matches), all used references, normalized source bars, and immutable setup/input snapshots. Model predictions stay separate from human reviews and execution failures.

## Identity, serialization, resume

Canonical JSON (`serialization.canonical_dumps`): preserve list order, sort mapping keys, timestamps as UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ`, dates as `YYYY-MM-DD`, reject Infinity, map legitimate missing scalars to `null`. Bytes contribute `{ "$bytes_sha256", "$byte_length" }` so digests stay bounded.

- **Candidate id** is stable from ticker + session date + chart profile (`make_candidate_id`). It does not include OHLCV values.
- **Source digest** (`BarWindow.source_digest` / `CandidateInput.source_digest`) detects changed bar content.
- **Semantic digest** (`PreparedScan.semantic_digest`) covers profile, setup content digests, example metadata + raw digests, candidate ids / eligibility / features / source digests, skips, and the classification-relevant config (`model`, timeout, max output tokens, image detail, mode). Resume requires this digest to match the frozen run. Semantic changes require a new run.
- **Raw digest** (`PreparedScan.raw_digest`) covers original YAML bytes, example payload digests, candidate source digests, and global-filter YAML. Audit only. YAML whitespace that does not change `content_digest` does not by itself block resume.

`VisionScanner.resume(run_id, prepared=None)` loads frozen inputs from the store. If `prepared` is supplied and `semantic_digest` differs, raise `FrozenInputConflictError`. Do not resubmit candidate ids that `list_committed_candidate_ids` already returns. Source-digest drift against the frozen snapshot is `changed_input` for that candidate, never a silent rewrite.

## Chart profile and last-N policy

Display-only SMA: rolling mean of the close series actually plotted, `min_periods=period`. No extra warm-up bars from before the window. A period longer than `lookback_bars` stays on the profile, plots as unavailable, and is reported in `MaAvailability.available=False`.

Last-N: the derived parquet may hold more bars than the profile lookback; slice down. If it holds fewer bars than the lookback for the **whole** file, fail with `ns rebuild-last100 --window-size N`. Isolated tickers with fewer rows than lookback are skipped, not filled.

## Model output schema

Schema file: `src/screener_loader/vision/response_schema.json` (`RESPONSE_SCHEMA_NAME = vision_batch_classification`).

```
results: [{ candidate_id, assessments: [{ setup_id, verdict, match_strength, reason, violated_required_rule_ids, missing_evidence }] }]
```

The model must return **exactly one assessment for every eligible setup of every candidate in the batch**. Extra candidates or setups are invalid. Reordering by id is allowed; missing coverage is not.

| Field | Rule |
| --- | --- |
| `verdict` | `match` \| `no_match` \| `uncertain` |
| `match_strength` | ordinal 1–3 **only** when `verdict=match`; otherwise JSON null. Not a probability. |
| `reason` | concise observable evidence, target ≤ 160 characters (`REASON_MAX_CHARS`) |
| `violated_required_rule_ids` | subset of this snapshot's required rule ids; used when a required rule is observably broken (typically `no_match`) |
| `missing_evidence` | required evidence not visible in the chart/features; typically `uncertain`. Bounded to 16 items, 200 chars each |

Match-strength rubric (prompt + validation):

1. Weak/marginal: required rules appear present but sloppy or incomplete.
2. Solid: required rules clear; some preferred characteristics.
3. Strong/canonical: required plus preferred; textbook for the snapshot.

Verdict consistency:

- `match`: strength in {1,2,3}; `violated_required_rule_ids` empty.
- `no_match`: strength null; may list violated required rules; must not claim a match.
- `uncertain`: strength null; use when required evidence is missing or the chart is ambiguous. Do **not** guess `no_match`.

Matched badges in the UI derive only from `verdict=match`. Join ticker, features, and artifacts from local inputs; the model is not an identity authority.

An invalid batch attempt is **not** accepted as partial guessed answers. Semantic validation failure is `ErrorKind=invalid_output`, never `no_match`.

## Error taxonomy and status

`ErrorKind`: `auth`, `config`, `rate_limit`, `transport`, `provider`, `refusal`, `timeout`, `incomplete`, `oversize`, `invalid_output`, `unavailable_input`, `stale_snapshot`, `short_window`, `profile_conflict`, `market_cap_unavailable`, `cancelled`, `changed_input`, `run_locked`.

`refusal`, `timeout`, `invalid_output`, and `unavailable_input` **never** become `no_match`.

Candidate status: `pending` → `completed` | `error` | `skipped`. Terminal for a candidate is completed/error/skipped.

Run status:

| Status | Meaning |
| --- | --- |
| `running` | freeze done, work in progress |
| `completed` | every candidate is completed or skipped; no remaining errors |
| `partial` | stopped with a mix of completed/skipped and errors (retries exhausted, isolated failures) |
| `failed` | preflight/auth/config/global failure, or nothing usable committed |
| `cancelled` | cancel flag observed; in-flight attempts may still be journaled |
| `dry_run` | render/compile/estimate only; zero provider calls; no invented predictions |

Zero-candidate success: a valid prepared scan with no candidates and no fatal skips completes as `completed` (or `dry_run` in that mode). Counts are zeros, not failures.

Terminal accounting (`RunSummary`) tracks unique candidates separately from setup-match counts and from attempt counts. Known usage on failed attempts is retained and summed when present; missing usage stays null. Remote billing is **not** exactly-once across network/crash ambiguity.

Immutable run inputs: setup snapshots, examples, chart profile, model id, eligible setups per candidate, window identity. Changing any of those semantically requires a new run.

## Protocols

### `ScanSource.prepare(config: ScanConfig) -> PreparedScan`

Raises `MarketCapUnavailableError`, `ChartProfileConflictError`, `InsufficientLastNError`, `StaleSnapshotError`, `VisionError`/`FileNotFoundError` for missing last-N or universe. Does not call the model.

### `ChartRenderer.render(window, profile, *, title=None) -> ChartArtifact`

Translates bars/profile, delegates to `render_chart_png`, returns PNG bytes, actual pixel size, digest, MA availability, final session date. No data queries, no file writes.

`wrap_upload(example)` returns the original image bytes as an artifact.

### `RequestCompiler.compile(...) -> CompiledRequest`

Provider-independent blocks, strict schema, fingerprint, labeled estimates. Relevant setups only (those eligible for at least one candidate in the batch), **all** examples in frozen order. Oversize → `OversizeRequestError` (no silent dropping).

### `Classifier.classify(request) -> ClassificationAttempt`

One attempt, timeout, SDK retries disabled. Returns success or a normalized error **with** any known usage, response id, Retry-After, timing, and sanitized output. Does not retry.

### `VisionScanner.run(prepared, *, cancel=None) -> ScanOutcome`

`resume(run_id, prepared=None, *, cancel=None) -> ScanOutcome`

Accepts `PreparedScan` only. No downloads, no eligibility reimplementation, no real source adapter.

### `RunStore` — calls the runner actually makes

1. `create_run(prepared)` — persist frozen inputs and artifacts metadata; status `running` or `dry_run`.
2. `lock_run(run_id)` — cross-process exclusive context manager; `RunLockedError` if held.
3. `save_artifact(run_id, artifact)` — persist chart bytes **before** the request uses them; return `ArtifactRef` with a **relative** path (`artifacts/candidates/{candidate_id}.png`, `artifacts/examples/{setup_id}/{example_id}.png`).
4. `get_artifact_bytes(run_id, artifact_id)` — runner/compiler/client and UI all read stored bytes, not a side cache.
5. `journal_attempt(run_id, attempt)` — append-only attempt log (start and completion, including failures).
6. `recover_attempt(run_id, batch_fingerprint)` — latest **accepted** attempt for that fingerprint, used before a retry/resubmit.
7. `commit_batch(run_id, batch_id=..., results=..., attempt=...)` — **atomic** commit of validated candidate results. Idempotent per `candidate_id`. Coordinator serializes store writes.
8. `mark_candidates(run_id, results)` — skips/errors without assessments.
9. `list_committed_candidate_ids(run_id)` — resume set.
10. `list_candidate_results(run_id)` — for summary accounting.
11. `load_run` / `load_frozen_inputs` — resume and digest checks.
12. `update_status` / `finalize(run_id, summary)` — terminal status + counts.
13. `export_manifest(run_id)` — relative artifact paths, digests, status, usage; no secrets.

Recovery: prefer a stored accepted response for a batch fingerprint over a new provider call. After an interrupted network call the remote charge may still be unknown.

### `ResultReader` — Agent 2

- **One row per candidate**, never one row per setup assessment.
- `matched_setups` is every assessment with `verdict=match` (badges).
- Setup filter is **ANY** of the selected `setup_ids` (empty means all setups in the run).
- Strength filter/sort is setup-aware: use `match_strength` from match assessments whose `setup_id` is in the selected setup filter. If a strength bound is set, exclude candidates with no matching strength in that subset. Sort that strength (default descending), then **stable `candidate_id` ascending** as the tie-break.
- Pagination is 1-based `PageRequest`. `ResultPage` includes `has_next` / `has_prev` / `next_page` / `prev_page` and distinct `total_candidates` vs `total_setup_matches`.
- A candidate **can appear in both Matches and Uncertain views** when different assessments differ. Each view is a query with `verdicts=("match",)` or `verdicts=("uncertain",)` (and optional setup filter). The row still contains all assessments.
- Reviews are separate from model verdicts. `review_state` is `unreviewed` when no current review exists; otherwise the current judgment (`agree` / `disagree` / `unsure`).

### `ReviewStore`

Append-only. `add_review` inserts a new `ReviewRecord` whose `supersedes_review_id` points at the previous current review for `(run_id, candidate_id, setup_id)`. Never mutate prior rows. `setup_id=None` is a whole-candidate review, distinct from per-setup reviews.

## Runner behavior (Agent 1)

- Validate prepared inputs, profile, examples, and config before provider calls. Broken required catalog references (missing example bytes/windows) fail preflight. Candidate-specific render failures skip/error that candidate only.
- Freeze inputs / `create_run` before API work. Prepare reference images once. Render with bounded memory. Persist chart bytes before compile/classify.
- Deterministic batches: candidates sorted by `candidate_id`, packed up to `batch_size` (default 10) and image/byte budgets. Default `max_concurrency=2`. Do not enqueue thousands of full payloads; in-flight batches ≤ `max_in_flight_batches`.
- Retry/pacing is the runner's: bounded exponential backoff + jitter, honor `Retry-After`, injectable clock/random/sleep. Stop dispatch on `auth`/`config`. Limited retry then **split** (partition remaining uncommitted candidates) for `incomplete` / `invalid_output` / `oversize`. Child batches never duplicate committed work. Do not rephrase refusals in a loop; refusals are candidate `error`.
- Dry-run: render, compile, estimate; zero `Classifier.classify` calls; no fake assessments. Demo: caller injects a labeled fake classifier (`mode="demo"`, `synthetic=True` on the outcome). Neither is a silent missing-key fallback.

## Out of scope

Alerts, scheduling, streaming, backtesting, model training, automatic example promotion, hosted frontends, paid market-wide scans from this agent, catalog/schema/data refactors, default-branch merge.
