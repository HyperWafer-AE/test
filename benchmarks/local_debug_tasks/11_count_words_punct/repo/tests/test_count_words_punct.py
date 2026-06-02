from text_utils import count_words

def test_count_words_punct():
    assert count_words('one, two.') == 2
