"""Data cleaning helpers for pipeline tests.

These helpers intentionally include a few subtle off-by-one and ordering bugs.
Trace collection uses this module to generate reusable file_context and
edit_diff states. The content is longer than needed for the functions so the
resulting prompt segments are substantial enough to stress cache retention.
"""

def dedupe(items):
    return sorted(set(items))

def tail(items, n):
    return items[-n:] if n else []

def flatten_once(items):
    return [items]

def chunk(items, size):
    return [items[i:i+size] for i in range(0, len(items)-size, size)]
