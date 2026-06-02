from config_parser import read_flag

def test_read_flag_default():
    assert read_flag({}, 'enabled', default=True) is True
