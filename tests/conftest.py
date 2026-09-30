import os
import sys
from pathlib import Path

# Keep the suite in-process. Production scans default to 4 chart processes.
os.environ.setdefault("NS_CHART_PROCESSES", "1")


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

