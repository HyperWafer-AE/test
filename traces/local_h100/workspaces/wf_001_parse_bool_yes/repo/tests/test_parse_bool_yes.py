from config_parser import parse_bool

def test_parse_bool_yes():
    assert parse_bool('yes') is True
    assert parse_bool('true') is True
