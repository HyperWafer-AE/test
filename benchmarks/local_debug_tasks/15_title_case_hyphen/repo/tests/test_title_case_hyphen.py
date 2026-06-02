from text_utils import title_words

def test_title_hyphen():
    assert title_words('hello-world') == 'Hello World'
