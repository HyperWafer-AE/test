from stats import mode

def test_mode_tie():
    assert mode(['a','b','a','b']) == 'a'
