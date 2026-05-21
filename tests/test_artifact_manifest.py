from __future__ import annotations

import json
from pathlib import Path


def test_artifact_manifest_schema_if_exported() -> None:
    path = Path("results/kvring_round2_paper_artifacts/artifact_manifest.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert "outputs" in data
    if data["outputs"]:
        first = data["outputs"][0]
        for key in ["file_path", "rows", "sha256", "command", "commit_hash"]:
            assert key in first
