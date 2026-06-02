from data_cleaner import chunk

def test_chunk_exact():
    assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]
