from __future__ import annotations

from src.normalize.command_parser import classify_phase, semantic_tool_from_command


def test_bash_command_semantic_tool_examples() -> None:
    cases = [
        ("pytest tests/test_x.py", "pytest", "execute/test"),
        ("grep -R foo src", "grep", "explore/read"),
        ("sed -i s/a/b/g file.py", "sed", "edit/write"),
        ("curl https://example.com", "curl", "retrieve/browser"),
    ]
    for command, tool, phase in cases:
        semantic = semantic_tool_from_command(command, tool_wrapper="bash_command")
        assert semantic == tool
        assert classify_phase(command, semantic, "bash_command") == phase

