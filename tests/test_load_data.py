from src.load_data import safe_key

def test_safe_key_valid_int():
    assert safe_key("123") == 123

def test_safe_key_valid_float():
    assert safe_key("123.0") == 123

def test_safe_key_invalid():
    assert safe_key("abc") == "abc"
