from __future__ import annotations

from src.normalize.normalizer import _extract_paths, normalize_rows


def test_html_path_not_truncated() -> None:
    paths = _extract_paths("created /app/out.html and app/main.py")
    assert "/app/out.html" in paths
    assert "/app/out.h" not in paths


def test_read_observation_paths_are_not_write() -> None:
    row = {
        "trial_id": "read-paths",
        "task_name": "read",
        "agent": "agent",
        "model": "model",
        "reward": 1,
        "steps": [
            {
                "src": "agent",
                "msg": "list files",
                "tools": [{"fn": "bash_command", "cmd": "ls"}],
                "obs": "foo.py\nbar.py\n/app/out.html",
            }
        ],
    }
    _, steps, objects, warnings = normalize_rows("terminalbench", [row])
    assert warnings == []
    assert steps["semantic_tool"].iloc[0] == "ls"
    files = objects[objects["object_type"] == "file"]
    assert not files.empty
    assert set(files["access_type"]).issubset({"mention", "read"})
    assert "write" not in set(files["access_type"])


def test_explicit_write_commands_mark_message_paths_as_write() -> None:
    row = {
        "trial_id": "write-paths",
        "task_name": "write",
        "agent": "agent",
        "model": "model",
        "reward": 1,
        "steps": [
            {
                "src": "agent",
                "msg": "patch file",
                "tools": [{"fn": "bash_command", "cmd": "sed -i s/a/b/g file.py"}],
                "obs": "done",
            },
                {
                    "src": "agent",
                    "msg": "patch",
                    "tools": [{"fn": "bash_command", "cmd": "write_file fix.py"}],
                    "obs": "patched fix.py",
                },
        ],
    }
    _, _, objects, _ = normalize_rows("terminalbench", [row])
    file_access = objects[objects["object_id"].isin(["file:file.py", "file:fix.py"])]
    assert not file_access.empty
    assert "write" in set(file_access["access_type"])
