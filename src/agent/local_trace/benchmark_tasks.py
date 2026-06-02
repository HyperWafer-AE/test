import json
from pathlib import Path


TASK_SPECS = [
    ("clamp_bounds", "math_utils.py", "clamp should keep values inside inclusive bounds", "return value", "return max(lower, min(upper, value))"),
    ("parse_bool_yes", "config_parser.py", "parse_bool should accept yes/no values", "return value.lower() in ('true', '1')", "return value.lower() in ('true', '1', 'yes', 'y', 'on')"),
    ("mean_empty", "stats.py", "mean should return 0 for empty inputs", "return sum(values) / len(values)", "return 0 if not values else sum(values) / len(values)"),
    ("dedupe_order", "data_cleaner.py", "dedupe should preserve first-seen order", "return sorted(set(items))", "return list(dict.fromkeys(items))"),
    ("slug_spaces", "text_utils.py", "slugify should collapse repeated whitespace", "return text.lower().replace(' ', '-')", "return '-'.join(text.lower().split())"),
    ("safe_divide_zero", "math_utils.py", "safe_divide should return default on zero divisor", "return numerator / denominator", "return default if denominator == 0 else numerator / denominator"),
    ("parse_int_blank", "config_parser.py", "parse_int should use default for blank strings", "return int(value)", "return default if str(value).strip() == '' else int(value)"),
    ("window_tail", "data_cleaner.py", "tail should handle n larger than sequence length", "return items[len(items)-n:len(items)-1]", "return items[-n:] if n else []"),
    ("normalize_none", "text_utils.py", "normalize should handle None as empty string", "return value.strip().lower()", "return '' if value is None else value.strip().lower()"),
    ("median_odd", "stats.py", "median should return middle element for odd-length sorted values", "return values[mid + 1]", "return values[mid]"),
    ("merge_config", "config_parser.py", "merge_config should let overrides win", "return {**override, **base}", "return {**base, **override}"),
    ("count_words_punct", "text_utils.py", "count_words should ignore punctuation", "return len(text.split())", "return len([w for w in text.replace(',', ' ').replace('.', ' ').split() if w])"),
    ("flatten_once", "data_cleaner.py", "flatten_once should flatten one nesting level", "return [items]", "return [x for group in items for x in group]"),
    ("variance_single", "stats.py", "variance should be 0 for a single item", "return sum((x - avg) ** 2 for x in values) / (len(values) - 1)", "return 0 if len(values) < 2 else sum((x - avg) ** 2 for x in values) / (len(values) - 1)"),
    ("env_list_trim", "config_parser.py", "parse_list should trim whitespace and drop empty parts", "return value.split(',')", "return [part.strip() for part in value.split(',') if part.strip()]"),
    ("title_case_hyphen", "text_utils.py", "title_words should treat hyphen as separator", "return ' '.join(w.capitalize() for w in text.split())", "return ' '.join(w.capitalize() for w in text.replace('-', ' ').split())"),
    ("chunk_exact", "data_cleaner.py", "chunk should include exact final chunk", "return [items[i:i+size] for i in range(0, len(items)-size, size)]", "return [items[i:i+size] for i in range(0, len(items), size)]"),
    ("percent_zero", "math_utils.py", "percent should return 0 when total is 0", "return part / total * 100", "return 0 if total == 0 else part / total * 100"),
    ("mode_tie", "stats.py", "mode should choose earliest value on tie", "return sorted(counts, key=counts.get)[-1]", "return max(counts, key=lambda x: (counts[x], -values.index(x)))"),
    ("json_flag", "config_parser.py", "read_flag should support missing key default", "return data[name]", "return data.get(name, default)"),
]


MODULE_BODIES = {
    "math_utils.py": """
\"\"\"Small numerical helpers.

The long module comments are intentional: local trace collection needs
file_context states large enough to create meaningful KV residency pressure.
These helpers are deliberately simple, but each function sits in a realistic
module with repeated explanatory context, edge-case descriptions, and examples.
Agents should read the file, identify the faulty behavior, patch a small line,
and run tests. The trace should preserve this file state for delayed reuse.
\"\"\"

def clamp(value, lower, upper):
    return value

def safe_divide(numerator, denominator, default=0):
    return numerator / denominator

def percent(part, total):
    return part / total * 100
""",
    "config_parser.py": """
\"\"\"Configuration parsing helpers used by command-line tools.

This file is intentionally verbose enough for Agent-on-Wafer state traces.
It contains repeated descriptions of configuration semantics: user overrides
should win, blank values should be safe, and optional flags should fall back
to defaults. Coding agents should use the test failure and this file_context
state across several turns rather than carrying a full message transcript.
\"\"\"

def parse_bool(value):
    return value.lower() in ('true', '1')

def parse_int(value, default=0):
    return int(value)

def merge_config(base, override):
    return {**override, **base}

def parse_list(value):
    return value.split(',')

def read_flag(data, name, default=False):
    return data[name]
""",
    "stats.py": """
\"\"\"Statistics helpers for small data processing tasks.

The implementation is tiny but the file contains enough descriptive material
for trace state construction. Tests exercise boundary cases such as empty
collections, single-item variance, odd medians, and deterministic tie-breaking.
The agent should create file_context, test_failure_summary, edit_diff, and
summary_state nodes while moving through a realistic repair loop.
\"\"\"

def mean(values):
    return sum(values) / len(values)

def median(values):
    values = sorted(values)
    mid = len(values) // 2
    return values[mid + 1]

def variance(values):
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / (len(values) - 1)

def mode(values):
    counts = {value: values.count(value) for value in values}
    return sorted(counts, key=counts.get)[-1]
""",
    "data_cleaner.py": """
\"\"\"Data cleaning helpers for pipeline tests.

These helpers intentionally include a few subtle off-by-one and ordering bugs.
Trace collection uses this module to generate reusable file_context and
edit_diff states. The content is longer than needed for the functions so the
resulting prompt segments are substantial enough to stress cache retention.
\"\"\"

def dedupe(items):
    return sorted(set(items))

def tail(items, n):
    return items[len(items)-n:len(items)-1]

def flatten_once(items):
    return [items]

def chunk(items, size):
    return [items[i:i+size] for i in range(0, len(items)-size, size)]
""",
    "text_utils.py": """
\"\"\"Text normalization helpers.

This file creates realistic search/read/edit behavior for local coding-agent
trace collection. The comments repeat the semantic intent of each function:
normalize whitespace, handle missing values, count robustly, and avoid
full-history prompt reconstruction. File states from this module are reused
after several tool and LLM turns to exercise delayed state residency.
\"\"\"

def slugify(text):
    return text.lower().replace(' ', '-')

def normalize(value):
    return value.strip().lower()

def count_words(text):
    return len(text.split())

def title_words(text):
    return ' '.join(w.capitalize() for w in text.split())
""",
}


TEST_TEMPLATES = {
    "clamp_bounds": "from math_utils import clamp\n\ndef test_clamp_bounds():\n    assert clamp(12, 0, 10) == 10\n    assert clamp(-1, 0, 10) == 0\n",
    "parse_bool_yes": "from config_parser import parse_bool\n\ndef test_parse_bool_yes():\n    assert parse_bool('yes') is True\n    assert parse_bool('true') is True\n",
    "mean_empty": "from stats import mean\n\ndef test_mean_empty():\n    assert mean([]) == 0\n",
    "dedupe_order": "from data_cleaner import dedupe\n\ndef test_dedupe_order():\n    assert dedupe(['b','a','b']) == ['b','a']\n",
    "slug_spaces": "from text_utils import slugify\n\ndef test_slug_spaces():\n    assert slugify('Hello   World') == 'hello-world'\n",
    "safe_divide_zero": "from math_utils import safe_divide\n\ndef test_safe_divide_zero():\n    assert safe_divide(5, 0, default=-1) == -1\n",
    "parse_int_blank": "from config_parser import parse_int\n\ndef test_parse_int_blank():\n    assert parse_int('', default=7) == 7\n",
    "window_tail": "from data_cleaner import tail\n\ndef test_tail_large_n():\n    assert tail([1,2,3], 9) == [1,2,3]\n",
    "normalize_none": "from text_utils import normalize\n\ndef test_normalize_none():\n    assert normalize(None) == ''\n",
    "median_odd": "from stats import median\n\ndef test_median_odd():\n    assert median([3,1,2]) == 2\n",
    "merge_config": "from config_parser import merge_config\n\ndef test_merge_config():\n    assert merge_config({'a':1,'b':2}, {'b':9}) == {'a':1,'b':9}\n",
    "count_words_punct": "from text_utils import count_words\n\ndef test_count_words_punct():\n    assert count_words('one, two.') == 2\n",
    "flatten_once": "from data_cleaner import flatten_once\n\ndef test_flatten_once():\n    assert flatten_once([[1,2],[3]]) == [1,2,3]\n",
    "variance_single": "from stats import variance\n\ndef test_variance_single():\n    assert variance([5]) == 0\n",
    "env_list_trim": "from config_parser import parse_list\n\ndef test_parse_list_trim():\n    assert parse_list('a, b,,c ') == ['a','b','c']\n",
    "title_case_hyphen": "from text_utils import title_words\n\ndef test_title_hyphen():\n    assert title_words('hello-world') == 'Hello World'\n",
    "chunk_exact": "from data_cleaner import chunk\n\ndef test_chunk_exact():\n    assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]\n",
    "percent_zero": "from math_utils import percent\n\ndef test_percent_zero():\n    assert percent(2, 0) == 0\n",
    "mode_tie": "from stats import mode\n\ndef test_mode_tie():\n    assert mode(['a','b','a','b']) == 'a'\n",
    "json_flag": "from config_parser import read_flag\n\ndef test_read_flag_default():\n    assert read_flag({}, 'enabled', default=True) is True\n",
}


def ensure_local_debug_tasks(root: str = "benchmarks/local_debug_tasks") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, (task_id, module, issue, old, new) in enumerate(TASK_SPECS):
        task_dir = root_path / f"{idx:02d}_{task_id}"
        repo_dir = task_dir / "repo"
        tests_dir = repo_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        for name, body in MODULE_BODIES.items():
            (repo_dir / name).write_text(body.strip() + "\n", encoding="utf-8")
        (tests_dir / f"test_{task_id}.py").write_text(TEST_TEMPLATES[task_id], encoding="utf-8")
        issue_text = f"Bug report: {issue}.\n\nRun `python -m pytest -q` and patch `{module}` so the test passes."
        (task_dir / "issue.md").write_text(issue_text, encoding="utf-8")
        metadata = {
            "task_id": task_id,
            "module": module,
            "issue": issue,
            "expected_command": "python -m pytest -q",
            "patch": {"path": module, "old": old, "new": new},
        }
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append({"task_id": task_id, "path": str(task_dir), **metadata})
    (root_path / "manifest.json").write_text(json.dumps({"tasks": manifest}, indent=2), encoding="utf-8")
    return root_path / "manifest.json"


if __name__ == "__main__":
    print(ensure_local_debug_tasks())

