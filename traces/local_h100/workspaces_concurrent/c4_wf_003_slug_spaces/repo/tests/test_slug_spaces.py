from text_utils import slugify

def test_slug_spaces():
    assert slugify('Hello   World') == 'hello-world'
