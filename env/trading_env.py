import random
from enum import Enum
import gymnasium as gym
import numpy as np
import pandas as pd
import pandas_ta as ta
from gymnasium import spaces
from stockholm import Money, Rate

from .utils import list_to_box_dict
from .data_source import TradingDataSource

class Action(Enum):
    Nothing = 0
    Sell = 1
    Buy = 2

    def opposite(self):
        return Action.Sell if self == Action.Buy else Action.Buy


class Position(Enum):
    Null = 0
    Long = 1
    Short = 2

    def opposite(self):
        return Position.Long if self == Position.Short else Position.Short


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_source: TradingDataSource, strategy: ta.Strategy):

        self.__starting_equity = Money(random.randrange(1000, 1500))
        print(f"Starting with {self.__starting_equity}$")

        self.__current_equity = self.__starting_equity
        self.__total_profit = Money(0)

        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short,
            "leverage": Rate(30)
        }

        self.__data_source = data_source
        self.__dataframe.ta.strategy(strategy)
        self.__dataframe = self.__dataframe.dropna(how="any", axis=0)
        self.__current_row = 0

        # spaces
        self.action_space = spaces.Discrete(len(Action))  # 0, 1 and 2

        self.indicators = strategy.ta
        self.observation_space = spaces.Dict(
            {
                "pl": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "pl_percent": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "position_type": spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            } | list_to_box_dict(list(self.__dataframe.columns))
        )

    def __get_pl(self) -> Money:
        if self.__stock["qty"] == 0 or self.__stock["type"] == Position.Null:
            return Money(0)
        if self.__stock["type"] == Position.Long:
            return (self.__get_current_stock_price() - self.__stock["avg_price_per_stock"]) * self.__stock["qty"] * \
                self.__stock["leverage"]
        else:
            return (self.__stock["avg_price_per_stock"] - self.__get_current_stock_price()) * self.__stock["qty"] * \
                self.__stock["leverage"]

    def __get_pl_rate(self) -> Rate:
        if self.__stock["qty"] == 0 or self.__stock["type"] == Position.Null:
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

        pl_percentage = self.__get_pl_rate()

        profit = self.__get_pl()

        if self.__stock["qty"] == 0:
            self.__stock["avg_price_per_stock"] = Money(0)
        self.__current_equity += profit

        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Null,
            "leverage": Rate(30)
        }
        print(f"{'+' if pl_percentage >= 0 else ''}{pl_percentage*100}%")
        return pl_percentage.as_float() * 100

    def _get_obs(self):

        row = self.__dataframe.iloc[self.__current_row]
        obs = {
                  "pl": 0 if self.__stock["qty"] == 0 else self.__get_pl().as_float(),
                  "pl_percent": 0 if self.__stock["qty"] == 0 else self.__get_pl_rate().as_float(),
                  "position_type": Position.Null if self.__stock["qty"] == 0 else self.__stock["type"].value,
              } | row.to_dict()

        for i, key in enumerate(obs.keys()):
            obs[key] = np.array([obs[key]], dtype=np.float64)
        return obs

    def _get_info(self):
        return {
            "total_pl": ((self.__get_current_equity() - self.__starting_equity) / self.__starting_equity).as_float()
        }

    def total_steps(self):
        return len(self.__dataframe.index)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.__dataframe = self.__data_source.next_data_batch()
        self.__current_equity = self.__starting_equity
        self.__current_row = 0
        self.__stock = {
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Null,
            "leverage": Rate(0)
        }
        print("reset() called")
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = Action(action)

        if self.__get_current_equity() <= 0:
            print("Early quitting")
            return self._get_obs(), -10, True, True, self._get_info()

        reward = .0

        # In case 30% of the value was lost
        if self.__get_pl_rate() <= -0.3:
            reward += self._close_positions()
        else:
            if action == Action.Buy:
                if self.__stock["type"] == Position.Short and self.__stock["qty"] != 0:
                    reward += self._close_positions()
                else:
                    self._buy(Rate(0.02))
                    reward += 0.2

            elif action == Action.Sell:
                if self.__stock["type"] == Position.Long and self.__stock["qty"] != 0:
                    reward += self._close_positions()
                else:
                    self._sell(Rate(0.02))
                    reward += 0.2

            self.__current_row += 1
        terminated = self.__current_row == len(self.__dataframe.index) - 1
        if terminated:
            reward += self._close_positions()
            print(f"Start: {self.__starting_equity}$, end: {self.__get_current_equity()}$")

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self, mode='human'):
        pass
