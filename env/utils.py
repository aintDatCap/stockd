from stockholm import Money
import pandas_ta as ta
import pandas as pd

def to_money(value) -> Money:
    if not isinstance(value, Money):
        return Money(value)
    return value


def generate_ta(df: pd.DataFrame):
    df.ta.sma(length=15)