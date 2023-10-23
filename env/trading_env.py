import random
from enum import Enum
from sys import float_info

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stockholm import Money, Rate


class Action(Enum):
    Nothing = 0
    Sell = 1
    Buy = 2

    def opposite(self):
        return Action.Sell if self == Action.Buy else Action.Buy


class Position(Enum):
    Long = 0
    Short = 0

    def opposite(self):
        return Position.Long if self == Position.Short else Position.Short


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataframe: pd.DataFrame):

        self.__starting_equity = Money(random.randrange(1000, 3000))
        print(f"Starting with {self.__starting_equity}$")

        self.__current_equity = self.__starting_equity
        self.__total_profit = Money(0)

        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short,
            "leverage": Rate(0)
        }

        self.__dataframe = dataframe  # pd.read_csv(self.__file)
        self.__current_row = 0

        # spaces
        self.action_space = spaces.MultiDiscrete(
            np.array([len(Action)]))  # 0 ... 10

        self.observation_space = spaces.Dict(
            {
                "open": spaces.Box(0, float_info.max, dtype=float),
                "high": spaces.Box(0, float_info.max, dtype=float),
                "low": spaces.Box(0, float_info.max, dtype=float),
                "close": spaces.Box(0, float_info.max, dtype=float),
                "volume": spaces.Box(0, float_info.max, dtype=float),
                "pl_percentage": spaces.Box(-1, float_info.max, dtype=float),
                "sma": spaces.Box(0, float_info.max, dtype=float),
                "williams_r": spaces.Box(-100, 0, dtype=float),
                "upper_bbands": spaces.Box(0, float_info.max, dtype=float),
                "middle_bbands": spaces.Box(0, float_info.max, dtype=float),
                "lower_bbands": spaces.Box(0, float_info.max, dtype=float),
                "avrg_true_range": spaces.Box(0, 1, dtype=float),
                "money_flow_idx": spaces.Box(0, 100, dtype=float),
                "rsi": spaces.Box(0, 100, dtype=float),
                "a/d": spaces.Box(0, float_info.max, dtype=float),
            }
        )

        self.__consequential_nothings = 0

    def __get_total_borrowable_money(self):
        return self.__current_equity * 0.8

    def __get_pl(self) -> Money:
        if self.__stock["qty"] == 0:
            return Money(0)
        if self.__stock["type"] == Position.Long:
            return (self.__get_current_stock_price() - self.__stock["avg_price_per_stock"]) * self.__stock["qty"] * \
                self.__stock["leverage"]
        else:
            return (self.__stock["avg_price_per_stock"] - self.__get_current_stock_price()) * self.__stock["qty"] * \
                self.__stock["leverage"]

    def __get_pl_percentage(self) -> Rate:
        if self.__stock["qty"] == 0:
            return Rate(0)
        if self.__stock["type"] == Position.Long:
            return Rate((self.__get_current_stock_price() - self.__stock["avg_price_per_stock"]) / self.__stock[
                "avg_price_per_stock"]) * self.__stock["leverage"]
        else:
            return Rate((self.__stock["avg_price_per_stock"] - self.__get_current_stock_price()) / self.__stock[
                "avg_price_per_stock"]) * self.__stock["leverage"]

    def __get_current_stock_price(self) -> Money:
        return Money(self.__dataframe.iloc[self.__current_row]["close"])

    def __get_current_equity(self) -> Money:
        return self.__current_equity + self.__get_pl()

    def __get_remaining_cash(self) -> Money:
        return max(self.__current_equity - self.__stock["avg_price_per_stock"] * abs(self.__stock["qty"]),
                   Money(0))

    def _buy(self, equity_rate: Rate):
        if self.__stock["qty"] > 0:
            return

        quantity = self.__get_current_equity() * equity_rate / self.__get_current_stock_price()

        self.__stock["avg_price_per_stock"] = self.__get_current_stock_price()
        self.__stock["qty"] = Rate(quantity)
        self.__stock["type"] = Position.Long
        self.__stock["leverage"] = Rate(30)

    def _sell(self, equity_rate: Rate):
        if self.__stock["qty"] > 0:
            return

        quantity = self.__get_current_equity() * equity_rate / self.__get_current_stock_price()

        self.__stock["avg_price_per_stock"] = self.__get_current_stock_price()
        self.__stock["qty"] = Rate(quantity)
        self.__stock["type"] = Position.Short
        self.__stock["leverage"] = Rate(30)

    def _close_positions(self) -> float:
        if self.__stock["qty"] == 0:
            return 0

        pl_percentage = self.__get_pl_percentage()

        profit = self.__get_pl()

        if self.__stock["qty"] == 0:
            self.__stock["avg_price_per_stock"] = Money(0)
        self.__current_equity += profit

        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short,
            "leverage": Rate(0)
        }
        return pl_percentage.as_float() * 100

    def _get_obs(self):
        dtypes = [np.float64] * 15

        row = self.__dataframe.iloc[self.__current_row]
        obs = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": self.__get_current_stock_price().as_float(),
            "volume": float(row["volume"]),
            "pl_percentage": self.__get_pl_percentage(),
            "sma": float(row["sma"]),
            "williams_r": float(row["williams_r"]),
            "upper_bbands": float(row["upper_bbands"]),
            "middle_bbands": float(row["middle_bbands"]),
            "lower_bbands": float(row["lower_bbands"]),
            "avrg_true_range": float(row["avrg_true_range"]),
            "money_flow_idx": float(row["money_flow_idx"]),
            "rsi": float(row["rsi"]),
            "a/d": float(row["a/d"]),

        }

        for i, key in enumerate(obs.keys()):
            obs[key] = np.array([obs[key]], dtype=dtypes[i])
        return obs

    def _get_info(self):
        return {
            "total_pl": ((self.__get_current_equity() - self.__starting_equity) / self.__starting_equity).as_float()
        }

    def total_steps(self):
        return len(self.__dataframe.index)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.__current_row = 0
        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short,
            "leverage": Rate(0)
        }

        return self._get_obs(), self._get_info()

    def step(self, action):

        _action = Action(action[0])

        if self.__get_current_equity() <= 0:
            print("Early quitting")
            return self._get_obs(), -10, True, True, self._get_info()

        reward = .0

        if _action == Action.Nothing:
            self.__consequential_nothings += 1
            if self.__consequential_nothings >= 390 * 20:  # 20 days
                reward -= 0.1
        else:
            self.__consequential_nothings = 0

        if _action == Action.Buy:
            if self.__stock["type"] == Position.Short and self.__stock["qty"] != 0:
                reward += self._close_positions()
            else:
                self._buy(Rate(0.02))

        elif _action == Action.Sell:
            if self.__stock["type"] == Position.Long and self.__stock["qty"] != 0:
                reward += self._close_positions()
            else:
                self._sell(Rate(0.02))

        self.__current_row += 1
        terminated = self.__current_row == len(self.__dataframe.index) - 1

        print(f"Start: {self.__starting_equity}$, end: {self.__get_current_equity()}$")
        self.reset()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self, mode='human'):
        pass
