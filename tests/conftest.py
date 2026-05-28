from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_ignore_collect(collection_path, config):  # type: ignore[no-untyped-def]
    path = Path(str(collection_path))
    if path.name == "test_flowmorph.py" and not (ROOT / "flowmorph").exists():
        return True
    return False
