"""Text normalization helpers.

This file creates realistic search/read/edit behavior for local coding-agent
trace collection. The comments repeat the semantic intent of each function:
normalize whitespace, handle missing values, count robustly, and avoid
full-history prompt reconstruction. File states from this module are reused
after several tool and LLM turns to exercise delayed state residency.
"""

def slugify(text):
    return text.lower().replace(' ', '-')

def normalize(value):
    return value.strip().lower()

def count_words(text):
    return len(text.split())

def title_words(text):
    return ' '.join(w.capitalize() for w in text.split())
