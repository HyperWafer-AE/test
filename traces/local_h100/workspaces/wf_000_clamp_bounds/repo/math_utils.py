"""Small numerical helpers.

The long module comments are intentional: local trace collection needs
file_context states large enough to create meaningful KV residency pressure.
These helpers are deliberately simple, but each function sits in a realistic
module with repeated explanatory context, edge-case descriptions, and examples.
Agents should read the file, identify the faulty behavior, patch a small line,
and run tests. The trace should preserve this file state for delayed reuse.
"""

def clamp(value, lower, upper):
    return max(lower, min(upper, value))

def safe_divide(numerator, denominator, default=0):
    return numerator / denominator

def percent(part, total):
    return part / total * 100
