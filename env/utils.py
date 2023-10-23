import numpy as np
import pandas_ta as ta
from gymnasium.spaces import Box
from stockholm import Money


def to_money(value) -> Money:
    if not isinstance(value, Money):
        return Money(value)
    return value


def strategy_to_dict(strategy: ta.Strategy) -> dict[str, Box]:
    result = {}
    indicators = sorted(strategy.ta, key=lambda x: x["kind"])

    for i, indicator in enumerate(indicators):
        result[f"{indicator['kind']}_{i}"] = Box(low=-np.inf, high=np.inf, dtype=np.float64)

    return result


def list_to_box_dict(elements: list[str]) -> dict[str, Box]:
    result = {}

    for element in elements:
        result[element] = Box(low=-np.inf, high=np.inf, dtype=np.float64)

    return result
