from stockholm import Money


def to_money(value) -> Money:
    if not isinstance(value, Money):
        return Money(value)
    return value
