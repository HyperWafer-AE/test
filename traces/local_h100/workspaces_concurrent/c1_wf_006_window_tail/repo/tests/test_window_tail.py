from data_cleaner import tail

def test_tail_large_n():
    assert tail([1,2,3], 9) == [1,2,3]
