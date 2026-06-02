from data_cleaner import dedupe

def test_dedupe_order():
    assert dedupe(['b','a','b']) == ['b','a']
