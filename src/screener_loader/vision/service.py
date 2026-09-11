"""Compose real ScanSource, Agent 1 pipeline, and Agent 2 store/query/review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import os

from ..config import LoaderConfig
from .adapters.demo import DEMO_MODEL_ID, DemoClassifier
from .adapters.source import SetupScanSource, vision_scan_root
from .protocols import CancelFlag
from .store import open_vision_scans
from .types import PreparedScan, ScanConfig, ScanOutcome, VisionError

Clock = Callable[[], Any]


def default_scan_root(loader: LoaderConfig) -> Path:
    return vision_scan_root(loader)


def require_explicit_live_model(model: str) -> str:
    value = str(model).strip()
    if not value:
        raise VisionError("Live scans require an explicit model id (--model or NS_VISION_MODEL)")
    if value in {DEMO_MODEL_ID, "dry-run-model", "demo-vision-classifier"}:
        raise VisionError("Live scans cannot use the demo/dry-run placeholder model id")
    return value


@dataclass
class _BoundImageLoader:
    store: Any
    run_id: str | None = None

    def __call__(self, artifact_id: str) -> bytes:
        if not self.run_id:
            raise KeyError(artifact_id)
        return self.store.get_artifact_bytes(self.run_id, artifact_id)


class _RunAwareStore:
    """Delegates to FilesystemRunStore while binding artifact fetches to the active run."""

    def __init__(self, store: Any, loader: _BoundImageLoader) -> None:
        self._store = store
        self._loader = loader

    def create_run(self, prepared: PreparedScan):
        stored = self._store.create_run(prepared)
        self._loader.run_id = stored.run_id
        return stored

    def lock_run(self, run_id: str):
        self._loader.run_id = run_id
        return self._store.lock_run(run_id)

    def __getattr__(self, name: str):
        return getattr(self._store, name)


class VisionApp:
    """Production composition. Demo/dry-run are explicit; missing keys never invent results."""

    def __init__(
        self,
        loader: LoaderConfig,
        *,
        clock: Clock | None = None,
        classifier: Any | None = None,
        renderer: Any | None = None,
        compiler: Any | None = None,
        scan_root: Path | None = None,
        max_candidates: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self.loader = loader
        self.source = SetupScanSource(loader, clock=clock, max_candidates=max_candidates)
        root = Path(scan_root) if scan_root is not None else default_scan_root(loader)
        root.mkdir(parents=True, exist_ok=True)
        self.scan_root = root
        store, reader, reviews = open_vision_scans(root)
        self.store = store
        self.reader = reader
        self.reviews = reviews
        self._classifier_override = classifier
        self._renderer = renderer
        self._compiler = compiler
        self._api_key = api_key

    def prepare(self, config: ScanConfig) -> PreparedScan:
        return self.source.prepare(config)

    def scanner_for(self, config: ScanConfig, prepared: PreparedScan | None = None):
        from .charts import MatplotlibChartRenderer
        from .prompts import SnapshotRequestCompiler
        from .scan import Scanner

        images = _BoundImageLoader(self.store)
        bound_store = _RunAwareStore(self.store, images)
        classifier = self._classifier_override
        if classifier is None:
            if config.mode == "demo":
                classifier = DemoClassifier.covering(prepared) if prepared is not None else DemoClassifier()
            else:
                if config.mode == "live":
                    require_explicit_live_model(config.model)
                    key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY")
                    if not key:
                        raise VisionError(
                            "OPENAI_API_KEY is missing; refusing to invent results. "
                            "Use --dry-run or --demo for offline checks."
                        )
                from .client import OpenAIClassifier

                classifier = OpenAIClassifier(api_key=self._api_key, get_image=images)
        renderer = self._renderer if self._renderer is not None else MatplotlibChartRenderer()
        compiler = self._compiler if self._compiler is not None else SnapshotRequestCompiler()
        return Scanner(
            renderer=renderer,
            compiler=compiler,
            classifier=classifier,
            store=bound_store,
        )

    def run(
        self,
        config: ScanConfig,
        *,
        cancel: CancelFlag | None = None,
        on_run_created: Callable[[str], None] | None = None,
    ) -> ScanOutcome:
        if config.mode == "live" and self._classifier_override is None:
            require_explicit_live_model(config.model)
            key = self._api_key if self._api_key is not None else os.environ.get("OPENAI_API_KEY")
            if not key:
                raise VisionError(
                    "OPENAI_API_KEY is missing; refusing to invent results. "
                    "Use --dry-run or --demo for offline checks."
                )
        prepared = self.prepare(config)
        scanner = self.scanner_for(config, prepared=prepared)
        self._bind_run_created(scanner, on_run_created)
        return scanner.run(prepared, cancel=cancel)

    def resume(
        self,
        run_id: str,
        prepared: PreparedScan | None = None,
        *,
        cancel: CancelFlag | None = None,
        on_run_created: Callable[[str], None] | None = None,
    ) -> ScanOutcome:
        if on_run_created is not None:
            on_run_created(str(run_id))
        frozen = self.store.load_frozen_inputs(run_id)
        scanner = self.scanner_for(frozen.config, prepared=frozen)
        self._bind_run_created(scanner, on_run_created)
        return scanner.resume(run_id, prepared, cancel=cancel)

    @staticmethod
    def _bind_run_created(scanner: Any, on_run_created: Callable[[str], None] | None) -> None:
        if on_run_created is None:
            return
        original = scanner.store.create_run

        def create_run(prepared: PreparedScan):
            stored = original(prepared)
            on_run_created(str(stored.run_id))
            return stored

        scanner.store.create_run = create_run  # type: ignore[method-assign]
