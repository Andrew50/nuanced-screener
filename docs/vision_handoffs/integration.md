# Agent 3 integration handoff

Started from foundation `54afa5bc258be6e80d1be3c22f43254a61c8386a` plus Agent 1 pipeline and Agent 2 store/UI already on this checkout. No fourth agent.

## Public symbols

| Symbol | Module | Role |
| --- | --- | --- |
| `SetupScanSource` | `vision.adapters.source` | Real `ScanSource.prepare` |
| `expected_completed_session` | `vision.adapters.freshness` | NYSE close / early-close cutoff |
| `DemoClassifier` | `vision.adapters.demo` | Explicit `mode=demo` only |
| `VisionApp` | `vision.service` | Source + renderer + compiler + store + scanner |
| `evaluate_run` / `evaluate_from_stores` | `vision.evaluate` | Offline multilabel counts |
| `ns screen` / `ns vision {scan,resume,runs,export,evaluate,view}` | `cli.vision` | Lazy optional imports |
| `ui.app:main` | `ui/app.py` | Shared Streamlit shell |

Scan root: `<repo>/data/vision_scans/` via `vision_scan_root` (does not edit `paths.py`).

## Dependencies

- Extra `vision = ["openai>=1.40", "matplotlib>=3.8"]` for headless scans.
- Extra `ui` unchanged (Streamlit + matplotlib).
- Package data: `screener_loader/vision/response_schema.json`.
- `.env.example`: `OPENAI_API_KEY`, `NS_VISION_MODEL` placeholders.
- CI installs `.[dev,ui]` so Streamlit/matplotlib tests run without OpenAI.
- No lockfile was present in the repo.

## Commands / tests actually run

```
ruff check .
python -m screener_loader --help
pytest -q
```

116 passed, 2 skipped. Base `ns --help` lists `vision` without importing OpenAI/Streamlit/matplotlib. Live OpenAI execution was not run (no billed scan).

## Integration notes

- `OpenAIClassifier.get_image` is bound to the active run via `_RunAwareStore`.
- Demo classifiers are built with `DemoClassifier.covering(prepared)` so mixed eligibility does not emit extra setups.
- `ns setups ui` now launches the shared app on the builder page; `ns vision view` selects Results (`NS_VISION_PAGE`, `NS_VISION_RUN_ID`). Results includes an explicit scan expander; `render_results_page` stays read-only.
- Production never falls back to `DemoClassifier` or `seed_synthetic_runs` when the API key is missing.

## Limitations

- Live classification against OpenAI is unverified here; missing credentials fail closed.
- Last-N provenance is `"unknown"` unless a uniform `source` column is present on the derived file. Current `LoaderConfig.polygon_adjusted` is not treated as historical proof.
- Filter-rejection reasons stay unavailable. Isolated short/stale windows are skips, not no-match.
- SMA display behavior is still the existing renderer; this agent only preserves the wrapper and profile.
