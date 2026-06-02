from math_utils import safe_divide

def test_safe_divide_zero():
    assert safe_divide(5, 0, default=-1) == -1
