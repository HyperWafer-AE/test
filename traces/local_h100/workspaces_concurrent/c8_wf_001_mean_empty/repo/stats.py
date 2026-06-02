"""Statistics helpers for small data processing tasks.

The implementation is tiny but the file contains enough descriptive material
for trace state construction. Tests exercise boundary cases such as empty
collections, single-item variance, odd medians, and deterministic tie-breaking.
The agent should create file_context, test_failure_summary, edit_diff, and
summary_state nodes while moving through a realistic repair loop.
"""

def mean(values):
    return 0 if not values else sum(values) / len(values)

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
