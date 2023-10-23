import pandas_ta as ta

default_strategy = ta.Strategy(
    name="Default Strategy",
    ta=[
        # SMA
        {"kind": "sma", "length": 1},
        {"kind": "sma", "length": 5},
        {"kind": "sma", "length": 10},
        {"kind": "sma", "length": 15},
        {"kind": "sma", "length": 30},
        {"kind": "sma", "length": 45},
        {"kind": "sma", "length": 60},
        {"kind": "sma", "length": 90},
        {"kind": "sma", "length": 120},
        {"kind": "sma", "length": 180},

        # WILLR
        {"kind": "willr", "length": 5},
        {"kind": "willr", "length": 10},
        {"kind": "willr", "length": 15},
        {"kind": "willr", "length": 30},
        {"kind": "willr", "length": 45},
        {"kind": "willr", "length": 60},
        {"kind": "willr", "length": 90},

        # DONCHIAN
        {"kind": "donchian", "lower_length": 20, "upper_length": 20},
        {"kind": "donchian", "lower_length": 20, "upper_length": 40},
        {"kind": "donchian", "lower_length": 20, "upper_length": 90},
        {"kind": "donchian", "lower_length": 20, "upper_length": 150},

        # AO
        {"kind": "ao"},

        # TRUE_RANGE
        {"kind": "true_range"},

        # STDEV
        {"kind": "stdev", "length": 10},
        {"kind": "stdev", "length": 15},
        {"kind": "stdev", "length": 30},
        {"kind": "stdev", "length": 45},
        {"kind": "stdev", "length": 60},
        {"kind": "stdev", "length": 120},

        # RSI
        {"kind": "rsi", "length": 5},
        {"kind": "rsi", "length": 10},
        {"kind": "rsi", "length": 15},
        {"kind": "rsi", "length": 30},

    ]
)
