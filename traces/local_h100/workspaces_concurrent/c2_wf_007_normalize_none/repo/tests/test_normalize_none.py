from text_utils import normalize

def test_normalize_none():
    assert normalize(None) == ''
