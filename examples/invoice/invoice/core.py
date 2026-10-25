"""A tiny invoicing module with one real (hand-written) bug, used as the quick-start example."""
from dataclasses import dataclass


@dataclass
class Line:
    sku: str
    qty: int
    unit_cents: int


def subtotal(lines):
    return sum(l.qty * l.unit_cents for l in lines)


def apply_discount(cents, percent):
    """Percent discount, rounded half up to the cent; never below zero."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return max(0, cents - (cents * percent + 50) // 100)


def paginate(items, page, per_page):
    """Return page `page` (1-based) of `items`."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be >= 1")
    start = page * per_page
    return items[start:start + per_page]


def total(lines, percent=0, tax_rate=0.1):
    net = apply_discount(subtotal(lines), percent)
    return net + round(net * tax_rate)
