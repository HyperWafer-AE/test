from config_parser import merge_config

def test_merge_config():
    assert merge_config({'a':1,'b':2}, {'b':9}) == {'a':1,'b':9}
