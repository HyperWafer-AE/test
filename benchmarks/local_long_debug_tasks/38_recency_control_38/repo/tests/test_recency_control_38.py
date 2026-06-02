from quick_math import clamp


def test_clamp_local_fix():
    assert clamp(12, 0, 10) == 10
    assert clamp(-1, 0, 10) == 0
