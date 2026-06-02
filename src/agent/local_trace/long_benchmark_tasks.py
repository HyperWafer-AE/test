import json
from pathlib import Path


OPPORTUNITY_FAMILIES = (
    ["multi_file_bug"] * 10
    + ["failure_loop"] * 8
    + ["large_context_bug"] * 7
    + ["config_or_data_pipeline_bug"] * 5
)
CONTROL_FAMILIES = ["recency_control"] * 10


def ensure_local_long_debug_tasks(root: str = "benchmarks/local_long_debug_tasks") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    tasks = []
    families = list(OPPORTUNITY_FAMILIES) + list(CONTROL_FAMILIES)
    for idx, family in enumerate(families):
        task_id = f"{family}_{idx:02d}"
        task_dir = root_path / f"{idx:02d}_{task_id}"
        repo_dir = task_dir / "repo"
        if family == "recency_control":
            metadata = write_control_task(task_id, family, task_dir, repo_dir, idx)
        else:
            metadata = write_opportunity_task(task_id, family, task_dir, repo_dir, idx)
        tasks.append({"task_id": task_id, "path": str(task_dir), **metadata})
    (root_path / "manifest.json").write_text(json.dumps({"tasks": tasks}, indent=2), encoding="utf-8")
    return root_path / "manifest.json"


def write_opportunity_task(task_id: str, family: str, task_dir: Path, repo_dir: Path, idx: int) -> dict:
    core = repo_dir / "core"
    tests = repo_dir / "tests"
    core.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    (core / "__init__.py").write_text("", encoding="utf-8")
    policy = {
        "threshold": 16 + idx % 3,
        "multiplier": 2 + idx % 2,
        "offset": 3 + idx % 4,
        "allow_negative": False,
    }
    value = 5 + idx % 4
    base = value + policy["offset"]
    expected_score = base * policy["multiplier"]
    if family == "failure_loop":
        policy["threshold"] = expected_score
    expected_label = "accept" if expected_score > policy["threshold"] else "review"
    first_files = ["core/config.py", "core/loader.py", "core/transform.py"]
    if family in {"failure_loop", "config_or_data_pipeline_bug"}:
        first_files.append("core/validator.py")
    if family == "large_context_bug":
        first_files.append("core/reporting.py")

    module_bodies = {
        "config.py": config_module(task_id, family, policy),
        "loader.py": loader_module(task_id, value),
        "transform.py": transform_module(task_id, family),
        "validator.py": validator_module(task_id, family),
        "reporting.py": reporting_module(task_id),
    }
    for name, body in module_bodies.items():
        (core / name).write_text(body, encoding="utf-8")

    diagnostic = diagnostic_blob(task_id, family, expected_score, expected_label)
    test_text = f"""
from core.config import load_policy
from core.loader import load_records
from core.transform import compute_score
from core.validator import classify_record


DIAGNOSTIC = {diagnostic!r}


def test_pipeline_score_and_classification():
    policy = load_policy("nightly")
    record = load_records()[0]
    score = compute_score(record, policy)
    label = classify_record(record, policy)
    assert score == {expected_score}, DIAGNOSTIC
    assert label == {expected_label!r}, DIAGNOSTIC
""".lstrip()
    (tests / f"test_{task_id}.py").write_text(test_text, encoding="utf-8")

    if family == "failure_loop":
        patch_path = "core/validator.py"
        old = "    return \"accept\" if score >= policy.get(\"threshold\", 0) else \"review\""
        wrong = "    return \"accept\""
        new = "    return \"accept\" if score > policy.get(\"threshold\", 0) else \"review\""
    elif family == "config_or_data_pipeline_bug":
        patch_path = "core/config.py"
        old = "    return {\"threshold\": 10, \"multiplier\": 1, \"offset\": DEFAULT_OFFSET, \"allow_negative\": True}"
        wrong = f"    return {{\"threshold\": {policy['threshold']}, \"multiplier\": 1, \"offset\": {policy['offset']}, \"allow_negative\": False}}"
        new = f"    return {{\"threshold\": {policy['threshold']}, \"multiplier\": {policy['multiplier']}, \"offset\": {policy['offset']}, \"allow_negative\": False}}"
    else:
        patch_path = "core/transform.py"
        old = "    return base  # BUG: multiplier from config is ignored"
        wrong = "    return base + policy.get(\"multiplier\", 1)  # BUG: adds multiplier instead of applying it"
        new = "    return base * policy.get(\"multiplier\", 1)"

    issue = f"""
The nightly data pipeline has a regression in task {task_id}.

Observed behavior:
* The test command `python -m pytest -q` fails.
* The relevant state is spread across configuration, loader, transform, and validator code.
* The first file read early should be revisited after the failed patch and failed test output.
* Do not solve this by carrying a full message history; use explicit state segments.

Expected behavior:
The pipeline should compute the configured score and classification for the synthetic record.
""".lstrip()
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "issue.md").write_text(issue, encoding="utf-8")
    metadata = {
        "task_id": task_id,
        "task_family": family,
        "expected_relevant_files": first_files,
        "expected_test_command": "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q",
        "expected_command": "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q",
        "expected_delayed_states": ["file_context", "test_failure_summary", "edit_diff"],
        "expected_min_steps": 18,
        "expected_large_state_tokens": {"file_context": 500, "test_failure_summary": 500},
        "expected_control_or_opportunity": "opportunity",
        "first_patch": {"path": patch_path, "old": old, "new": wrong},
        "final_patch": {"path": patch_path, "old": wrong, "new": new},
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_control_task(task_id: str, family: str, task_dir: Path, repo_dir: Path, idx: int) -> dict:
    tests = repo_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (repo_dir / "quick_math.py").write_text(
        """
def clamp(value, lower, upper):
    return value
""".lstrip(),
        encoding="utf-8",
    )
    (tests / f"test_{task_id}.py").write_text(
        """
from quick_math import clamp


def test_clamp_local_fix():
    assert clamp(12, 0, 10) == 10
    assert clamp(-1, 0, 10) == 0
""".lstrip(),
        encoding="utf-8",
    )
    issue = "Short recency-control bug: clamp should keep values inside inclusive bounds.\n"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "issue.md").write_text(issue, encoding="utf-8")
    metadata = {
        "task_id": task_id,
        "task_family": family,
        "expected_relevant_files": ["quick_math.py"],
        "expected_test_command": "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q",
        "expected_command": "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q",
        "expected_delayed_states": [],
        "expected_min_steps": 10,
        "expected_large_state_tokens": {},
        "expected_control_or_opportunity": "control",
        "first_patch": {
            "path": "quick_math.py",
            "old": "    return value",
            "new": "    return max(lower, value)",
        },
        "final_patch": {
            "path": "quick_math.py",
            "old": "    return max(lower, value)",
            "new": "    return max(lower, min(upper, value))",
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def large_comment(task_id: str, topic: str, paragraphs: int = 12) -> str:
    base = (
        f"{task_id} {topic}: This module participates in a multi-stage debugging workflow. "
        "The agent should preserve this state as a named file_context, drop it for several "
        "intervening LLM and tool turns, and then reintroduce it when the failed hypothesis "
        "requires checking early assumptions. The text intentionally describes invariants, "
        "edge cases, configuration semantics, validation behavior, and expected data flow. "
        "It is not a full-history message transcript; it is a reusable source artifact. "
    )
    return "\n".join(base for _ in range(paragraphs))


def config_module(task_id: str, family: str, policy: dict) -> str:
    if family == "config_or_data_pipeline_bug":
        policy_line = '    return {"threshold": 10, "multiplier": 1, "offset": DEFAULT_OFFSET, "allow_negative": True}'
    else:
        policy_line = f'    return {{"threshold": {policy["threshold"]}, "multiplier": {policy["multiplier"]}, "offset": {policy["offset"]}, "allow_negative": False}}'
    return f'''
"""Configuration layer.

{large_comment(task_id, "configuration", 14)}
"""

DEFAULT_OFFSET = {policy["offset"]}


def load_policy(name):
    # BUG in config_or_data_pipeline_bug tasks: multiplier may be ignored.
{policy_line}


def explain_policy(policy):
    return "threshold={{threshold}} multiplier={{multiplier}} offset={{offset}}".format(**policy)
'''.lstrip()


def loader_module(task_id: str, value: int) -> str:
    return f'''
"""Loader layer.

{large_comment(task_id, "loader", 13)}
"""


def load_records():
    return [
        {{"id": "{task_id}", "value": {value}, "source": "nightly", "quality": "checked"}},
        {{"id": "{task_id}-shadow", "value": {value + 1}, "source": "shadow", "quality": "ignored"}},
    ]


def describe_records(records):
    return [record["id"] + ":" + str(record["value"]) for record in records]
'''.lstrip()


def transform_module(task_id: str, family: str) -> str:
    if family in {"multi_file_bug", "large_context_bug"}:
        score_line = '    return base  # BUG: multiplier from config is ignored'
    else:
        score_line = '    return base * policy.get("multiplier", 1)'
    return f'''
"""Transform layer.

{large_comment(task_id, "transform", 15)}
"""


def compute_score(record, policy):
    value = record["value"]
    if value < 0 and not policy.get("allow_negative", False):
        value = 0
    base = value + policy.get("offset", 0)
{score_line}


def normalize_record(record):
    return {{"id": record["id"], "value": int(record["value"]), "source": record.get("source", "unknown")}}
'''.lstrip()


def validator_module(task_id: str, family: str) -> str:
    if family == "failure_loop":
        classify_line = '    return "accept" if score >= policy.get("threshold", 0) else "review"'
    else:
        classify_line = '    return "accept" if score > policy.get("threshold", 0) else "review"'
    return f'''
"""Validator layer.

{large_comment(task_id, "validator", 13)}
"""

from core.transform import compute_score


def classify_record(record, policy):
    score = compute_score(record, policy)
{classify_line}


def validation_notes(record, policy):
    return {{"id": record["id"], "threshold": policy.get("threshold", 0), "score": compute_score(record, policy)}}
'''.lstrip()


def reporting_module(task_id: str) -> str:
    return f'''
"""Reporting layer.

{large_comment(task_id, "reporting", 12)}
"""


def render_report(record, score, label):
    return f"{{record['id']}} score={{score}} label={{label}}"


def render_debug_table(rows):
    return "\\n".join(str(row) for row in rows)
'''.lstrip()


def diagnostic_blob(task_id: str, family: str, expected_score: int, expected_label: str) -> str:
    line = (
        f"{task_id} {family} diagnostic: expected score {expected_score} and label {expected_label}. "
        "The failure should preserve the earlier configuration and transform file_context states, "
        "the raw pytest failure, and the failed edit diff for delayed reuse analysis. "
    )
    return "\n".join(line for _ in range(80))


if __name__ == "__main__":
    print(ensure_local_long_debug_tasks())
