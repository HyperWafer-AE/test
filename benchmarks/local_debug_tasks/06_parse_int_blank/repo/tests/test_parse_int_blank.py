from config_parser import parse_int

def test_parse_int_blank():
    assert parse_int('', default=7) == 7
