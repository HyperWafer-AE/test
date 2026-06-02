from config_parser import parse_list

def test_parse_list_trim():
    assert parse_list('a, b,,c ') == ['a','b','c']
