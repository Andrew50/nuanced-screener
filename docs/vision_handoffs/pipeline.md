# Agent 1 pipeline handoff

Foundation checkpoint (Agents 2 and 3 start here): `54afa5bc258be6e80d1be3c22f43254a61c8386a`

This commit adds the chart wrapper, request compiler, one-attempt OpenAI client, batching, and resumable runner on top of that foundation.

## Public symbols

Import contracts from `screener_loader.vision` (no OpenAI/matplotlib at import time). Pipeline implementations:

| Symbol | Module | Role |
| --- | --- | --- |
| `ScanConfig`, `PreparedScan`, `ChartProfile`, query/review types | `vision.types` | Frozen models |
| `ScanSource`, `ChartRenderer`, `RequestCompiler`, `Classifier`, `RunStore`, `ResultReader`, `ReviewStore`, `VisionScanner` | `vision.protocols` | Interfaces |
| `snapshot_setup`, `resolve_chart_profile`, `freeze_prepared_scan`, `make_candidate_id`, `moving_average_availability` | `vision.snapshots` | Catalog bindings |
| `MatplotlibChartRenderer`, `render_window_png`, `png_dimensions` | `vision.charts` | Thin wrapper over `setups.charts` |
| `SnapshotRequestCompiler` | `vision.prompts` | Provider-independent compile |
| `validate_classification_payload` | `vision.validation` | Exact coverage; invalid ≠ partial |
| `OpenAIClassifier` | `vision.client` | One Responses API attempt, `max_retries=0` |
| `Scanner` / `VisionScanner` | `vision.scan` | Run / resume |
| `chunk_candidates`, `split_candidates`, `retry_delay_seconds` | `vision.batching` | Packing / backoff |
| `load_response_schema` | `vision.types` | `response_schema.json` |

Test fakes (not production): `tests/vision_support.py` — `FakeRenderer`, `FakeCompiler`, `FakeClassifier`, `InMemoryRunStore`, `FakeResultReader`.

## Dependencies for Agent 3

Do **not** add these from Agent 1's PR if you are splitting commits; record them here.

- `openai` (verified locally **3.8.0**) for `OpenAIClassifier`. Suggested extra: `vision = ["openai>=1.40"]` (or pin 3.8). Lazy-imported; base `ns` commands work without it.
- Matplotlib remains in the existing `ui` extra. Renderer still lazy-imports it.
- Package data: ship `screener_loader/vision/response_schema.json` (editable installs already see it via `__file__`).

`.env.example`: document `OPENAI_API_KEY` and an explicit model id. Missing key is `auth`, never a fake scan.

## Commands / tests actually run

```
ruff check src/screener_loader/vision src/screener_loader/setups/charts.py tests/test_vision_*.py tests/vision_support.py
pytest tests/test_vision_contract.py tests/test_vision_charts.py tests/test_vision_classification.py tests/test_vision_scan.py tests/test_chart_renderer.py tests/test_setup_service.py tests/test_setup_eligibility.py tests/test_cli_setups.py -q
python -m screener_loader --help
```

54 passed in that set. Visual: `tests/fixtures/vision/rendered_sample.png` (1011×611, tight bbox). Wrapper bytes equal `render_chart_png` for the same input/style/title. SMA 50 is unavailable on a 40-bar window and kept on the profile. `tests/fixtures/vision/synthetic.png` is the 8×6 store/UI stub.

No live API spend. Classification tests mock the SDK/transport.

## What Agent 2 needs

Implement `RunStore`, `ResultReader`, `ReviewStore` exactly as `docs/VISION_CONTRACT.md` and `vision.protocols`. Runner calls, in order:

1. `create_run(prepared)`
2. `lock_run(run_id)` (context manager)
3. `save_artifact` before compile/classify
4. `journal_attempt` for every provider attempt (including failures)
5. `recover_attempt(run_id, fingerprint)` before resubmit
6. `commit_batch` atomically for validated results (idempotent per `candidate_id`)
7. `mark_candidates` for skips/errors without assessments
8. `list_candidate_results` / `list_committed_candidate_ids` for resume
9. `finalize`

Relative artifact paths, run locks, append-only reviews, one row per candidate, ANY setup filter, setup-aware strength sort with `candidate_id` tie-break. Use `tests/fixtures/vision/synthetic.png` until real charts exist.

## What Agent 3 needs

- Real `ScanSource.prepare`: `SetupService` + `compute_eligibility` + bulk `last_100_bars_parquet` slice + `load_universe`. Intersect universe. Fail `MarketCapUnavailableError`. Global short last-N → `InsufficientLastNError` naming `ns rebuild-last100 --window-size N`. Isolated short/stale → `PreparedScan.skips`. Latest-session freshness only.
- Wire `Scanner(renderer=MatplotlibChartRenderer(), compiler=SnapshotRequestCompiler(), classifier=OpenAIClassifier(get_image=store.get_artifact_bytes), store=...)`.
- Dry-run and demo must be explicit (`ScanConfig.mode`). Demo injects `FakeClassifier` and stays labeled `synthetic`.
- CLI / Streamlit navigation / lockfile / README / CI extras.
- Compose Agents 1+2; no fourth agent.

## Limitations

- No real ScanSource, persistence, results UI, or CLI registration here.
- OpenAI billing is not exactly-once across crash/network ambiguity; recover stored JSON when present.
- `max_concurrency=16`, `batch_size=10` match the measured OpenAI headroom for this account (500 RPM / 500k TPM). The 3,000-candidate fake test still passes an explicit cap and guards unbounded in-flight compile/classify (`peak_in_flight <= 2`, `max_seen_candidates <= 10`).
- Refusal/timeout/invalid_output/unavailable_input never become `no_match`.
- SMA uses displayed bars only (`min_periods=period`); long periods stay configured and report unavailable.
- Production never silently substitutes `FakeClassifier` when the API key is missing.
