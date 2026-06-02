from data_cleaner import flatten_once

def test_flatten_once():
    assert flatten_once([[1,2],[3]]) == [1,2,3]
