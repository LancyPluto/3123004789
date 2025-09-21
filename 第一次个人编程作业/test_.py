import pytest
from work import (
    clean_text,
    tokenize,
    cosine_similarity,
    lcs_length,
    compute_duplicate_rate,
)

def test_clean_text_normal():
    text = "我喜欢，Python!!!"
    result = clean_text(text)
    assert "Python" in result
    assert "," not in result
    assert "!" not in result

def test_clean_text_empty():
    with pytest.raises(ValueError):
        clean_text("!!!")

def test_tokenize():
    text = "我爱自然语言处理"
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert "自然语言" in "".join(tokens)

def test_cosine_similarity_basic():
    a = ["我", "喜欢", "机器学习"]
    b = ["我", "热爱", "机器学习"]
    sim = cosine_similarity(a, b)
    assert 0 <= sim <= 1
    assert sim > 0

def test_lcs_length():
    a = ["A", "B", "C", "D"]
    b = ["B", "C", "E"]
    assert lcs_length(a, b) == 2  # B、C

def test_compute_duplicate_rate_identical():
    text = "深度学习改变世界"
    rate = compute_duplicate_rate(text, text)
    assert rate == 100.0

def test_compute_duplicate_rate_different():
    a = "机器学习"
    b = "天气晴朗"
    rate = compute_duplicate_rate(a, b)
    assert rate < 5

def test_compute_duplicate_rate_empty_orig():
    with pytest.raises(ZeroDivisionError):
        compute_duplicate_rate("", "任意内容")
