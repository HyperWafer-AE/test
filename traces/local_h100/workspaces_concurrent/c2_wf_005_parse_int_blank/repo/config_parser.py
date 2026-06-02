"""Configuration parsing helpers used by command-line tools.

This file is intentionally verbose enough for Agent-on-Wafer state traces.
It contains repeated descriptions of configuration semantics: user overrides
should win, blank values should be safe, and optional flags should fall back
to defaults. Coding agents should use the test failure and this file_context
state across several turns rather than carrying a full message transcript.
"""

def parse_bool(value):
    return value.lower() in ('true', '1')

def parse_int(value, default=0):
    return default if str(value).strip() == '' else int(value)

def merge_config(base, override):
    return {**override, **base}

def parse_list(value):
    return value.split(',')

def read_flag(data, name, default=False):
    return data[name]
