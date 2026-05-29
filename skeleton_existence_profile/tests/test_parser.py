from __future__ import annotations

from skeleton.parser import command_artifact_flag, has_explicit_write, semantic_tool_clean, semantic_tool_from_command


def test_semantic_parser_common_tools() -> None:
    assert semantic_tool_clean(semantic_tool_from_command("grep -R foo .")) == "search"
    assert semantic_tool_clean(semantic_tool_from_command("cat src/app.py")) == "read_file"
    assert semantic_tool_clean(semantic_tool_from_command("pytest tests/test_app.py")) == "test"
    assert semantic_tool_clean(semantic_tool_from_command("apply_patch <<'PATCH'\nPATCH")) == "edit"
    assert semantic_tool_clean(semantic_tool_from_command("search_dir foo src")) == "search"
    assert semantic_tool_clean(semantic_tool_from_command("open src/app.py")) == "read_file"


def test_sed_read_vs_edit() -> None:
    assert semantic_tool_clean(semantic_tool_from_command("sed -n '1,10p' src/app.py")) == "read_file"
    assert semantic_tool_clean(semantic_tool_from_command("sed -i 's/a/b/' src/app.py")) == "edit"


def test_echo_write_detection() -> None:
    assert semantic_tool_clean(semantic_tool_from_command("echo hello")) == "unknown"
    assert not has_explicit_write("echo hello", "echo")
    assert semantic_tool_clean(semantic_tool_from_command("echo hello > file.txt")) == "edit"
    assert has_explicit_write("echo hello > file.txt", "echo")


def test_command_artifact_detection() -> None:
    assert command_artifact_flag("{not a command}", "unknown")
    assert not command_artifact_flag("cat file.txt", "read_file")
