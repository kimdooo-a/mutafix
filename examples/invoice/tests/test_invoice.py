import pytest

from invoice import Line, apply_discount, paginate, subtotal, total

LINES = [Line("A", 2, 1250), Line("B", 1, 999), Line("C", 3, 100)]


def test_subtotal():
    assert subtotal(LINES) == 3799


def test_discount_rounds_half_up():
    assert apply_discount(3799, 10) == 3419


def test_discount_bounds():
    with pytest.raises(ValueError):
        apply_discount(100, 101)


def test_total_with_tax():
    assert total(LINES, percent=10) == 3761


def test_first_page():
    assert paginate(list(range(10)), 1, 4) == [0, 1, 2, 3]


def test_last_page_is_partial():
    assert paginate(list(range(10)), 3, 4) == [8, 9]
