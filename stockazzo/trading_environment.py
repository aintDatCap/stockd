import math
import os.path
import random
from datetime import datetime
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

    def __init__(self, file: str, symbol: str = ""):
        self.__symbol = symbol if symbol != "" else os.path.basename(file)
        self.__starting_equity = Money(random.randrange(200, 7000))
        print(f"Starting with {self.__starting_equity}$")

        self.__current_equity = self.__starting_equity

        self.__borrowed_money = Money(0)

        self.__intradays: list[datetime] = []
        self.__daily_action_type: list[Action] = []  # convert it in daily action
        self.__stock = {
            "symbol": symbol,
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short
        }

        self.__current_date: datetime = None
        self.__file = file
        self.__dataframe: pd.DataFrame = pd.read_csv(self.__file)
        self.__current_row = 0

        # spaces
        self.action_space = spaces.MultiDiscrete(
            np.array([len(Action), 11] + [25_000] * 5))  # 0 ... 10

        self.observation_space = spaces.Dict(
            {
                "price_per_stock": spaces.Box(0, float_info.max, dtype=float),
                "williams_r": spaces.Box(-100, float_info.max, dtype=float),
                "donchian_channels_top": spaces.Box(0, float_info.max, dtype=float),
                "donchian_channels_middle": spaces.Box(0, float_info.max, dtype=float),
                "donchian_channels_bottom": spaces.Box(0, float_info.max, dtype=float),
                "moving_average": spaces.Box(0, float_info.max, dtype=float),
                "accumulation_distribution": spaces.Box(0, 1, dtype=float),
                "relative_strength_index": spaces.Box(0, 100, dtype=float),
                "tenkan_sen": spaces.Box(0, float_info.max, dtype=float),
                "kijun_sen": spaces.Box(0, float_info.max, dtype=float),
                "chikou_span": spaces.Box(0, float_info.max, dtype=float),
                "senkou_span_a": spaces.Box(0, float_info.max, dtype=float),
                "senkou_span_b": spaces.Box(0, float_info.max, dtype=float)
            }
        )

        self.__consequential_nothings = 0
        self.__callback = [0] * 5

    def __get_remaining_intraday(self) -> int:
        if self.__current_equity < 26_000:
            return 3 - len(self.__intradays)
        return 3

    def __get_total_borrowable_money(self):
        return self.__current_equity * 0.8

    def __get_remaining_borrowable_money(self):
        return self.__get_total_borrowable_money() - self.__borrowed_money

    def __borrow_money(self, money_to_borrow: Money) -> bool:
        total_borrowable_money = self.__get_remaining_borrowable_money()

        if self.__borrowed_money + money_to_borrow <= total_borrowable_money:
            self.__borrowed_money += money_to_borrow
            return True
        return False

    def __get_pl(self) -> Money:
        if self.__stock["qty"] == 0:
            return Money(0)
        if self.__stock["type"] == Position.Long:
            return (self.__get_current_stock_price() - self.__stock["avg_price_per_stock"]) * self.__stock["qty"]
        else:
            return (self.__stock["avg_price_per_stock"] - self.__get_current_stock_price()) * self.__stock["qty"]

    def __get_pl_percentage(self) -> Rate:
        if self.__stock["qty"] == 0:
            return Rate(0)
        if self.__stock["type"] == Position.Long:
            return Rate((self.__get_current_stock_price() - self.__stock["avg_price_per_stock"]) / self.__stock[
                "avg_price_per_stock"])
        else:
            return Rate((self.__stock["avg_price_per_stock"] - self.__get_current_stock_price()) / self.__stock[
                "avg_price_per_stock"])

    def __get_current_stock_price(self) -> Money:
        return Money(self.__dataframe.iloc[self.__current_row]["Close"])

    def __get_current_equity(self) -> Money:
        return self.__current_equity + self.__get_pl()

    def __get_remaining_cash(self) -> Money:
        return max(self.__current_equity - self.__stock["avg_price_per_stock"] * abs(self.__stock["qty"]),
                   Money(0))

    def _new_day(self, new_date: datetime) -> float:
        # self.__current_equity -= (self.__borrowed_money * 0.075) / 360
        for i, intraday in enumerate(self.__intradays):
            if np.busday_count(intraday.date(), new_date.date()) >= 5:
                self.__intradays.pop(i)

        self.__daily_action_type = []
        print(f"{self.__symbol} Current equity: {self.__get_current_equity()}")
        # print(self._get_obs())
        return (self.__get_current_equity() / self.__starting_equity).as_float()

    def _buy(self, quantity: int):
        possessed_quantity = self.__stock["qty"]
        if possessed_quantity < 0 or quantity <= 0:
            return

        if self.__stock["type"] == Position.Short and self.__stock["qty"] != 0:
            return

        cash = self.__get_remaining_cash()
        total_price = self.__get_current_stock_price() * quantity

        if total_price > cash:
            money_to_borrow = total_price - cash
            successfully_borrowed = self.__borrow_money(money_to_borrow)
            if not successfully_borrowed:
                return

        self.__daily_action_type.append(Action.Buy)

        self.__stock["avg_price_per_stock"] = (self.__stock["avg_price_per_stock"] * self.__stock[
            "qty"] + self.__get_current_stock_price() * quantity) / (self.__stock["qty"] + quantity)
        self.__stock["qty"] += quantity
        self.__stock["type"] = Position.Long

    def _sell(self, quantity: int):
        possessed_quantity = self.__stock["qty"]
        if possessed_quantity < 0 or quantity <= 0:
            return

        if self.__stock["type"] == Position.Long and self.__stock["qty"] != 0:
            return

        if quantity + possessed_quantity >= 10:
            quantity = 10 - possessed_quantity

        cash = self.__get_remaining_cash()
        total_price = self.__get_current_stock_price() * quantity

        if total_price > cash:
            money_to_borrow = total_price - cash
            successfully_borrowed = self.__borrow_money(money_to_borrow)
            if not successfully_borrowed:
                return

        self.__daily_action_type.append(Action.Sell)

        self.__stock["avg_price_per_stock"] = (self.__stock["avg_price_per_stock"] * self.__stock[
            "qty"] + self.__get_current_stock_price() * quantity) / (self.__stock["qty"] + quantity)
        self.__stock["qty"] += quantity
        self.__stock["type"] = Position.Short

    def _close_positions(self, quantity: int) -> float:

        current_stock_price = self.__get_current_stock_price()
        possessed_quantity = self.__stock["qty"]

        if quantity > possessed_quantity:
            quantity = possessed_quantity

        if self.__get_pl_percentage() <= 0:
            return -0.1

        if len(self.__daily_action_type) > 0:
            if self.__get_remaining_intraday() <= 0:
                return 0
            self.__intradays.append(self.__current_date)
            self.__daily_action_type.pop()  # self.__daily_action_type.index(Action.Buy))

        if self.__stock["type"] == Position.Long:
            profit = (current_stock_price - self.__stock["avg_price_per_stock"]) * quantity
        else:
            profit = (self.__stock["avg_price_per_stock"] - current_stock_price) * quantity

        self._total_profit += profit

        if self.__borrowed_money > 0:
            self.__borrowed_money -= min(self.__borrowed_money, quantity * current_stock_price)
        self.__stock["qty"] -= quantity
        if self.__stock["qty"] == 0:
            self.__stock["avg_price_per_stock"] = Money(0)
        self.__current_equity += profit

        return profit * 10

    def _get_obs(self):
        dtypes = [np.float64] * 2 + [np.int32] + [np.float64] + [np.int64] + [np.float64] * 4 + [
            np.int32] * 3 + [np.int32]
        obs = {
            "current_price_per_stock": self.__get_current_stock_price().as_float(),
            "avg_price_per_stock": self.__stock["avg_price_per_stock"].as_float(),
            "possessed_qty": self.__stock["qty"].as_float(),
            "acquirable_qty": math.floor(((
                                                  self.__get_remaining_cash() + self.__get_remaining_borrowable_money()) / self.__get_current_stock_price()).as_float()),
            "position_type": self.__stock["type"].value,
            "pl": self.__get_pl().as_float(),
            "pl_percentage": self.__get_pl_percentage().as_float(),
            "remaining_cash": self.__get_remaining_cash().as_float(),
            "remaining_borrowable_money": self.__get_remaining_borrowable_money(),
            "day": self.__current_date.weekday(),
            "minute": int((self.__current_date - self.__current_date.replace(hour=9, minute=30)).total_seconds() / 60),
            "remaining_intradays": self.__get_remaining_intraday(),
            "callback": self.__callback
        }

        for i, key in enumerate(obs.keys()):
            obs[key] = np.array([obs[key]], dtype=dtypes[i])
        return obs

    def _get_info(self):
        return {
            "total_profit": self._total_profit.as_float(),
            "total_pl": ((self.__get_current_equity() - self.__starting_equity) / self.__starting_equity).as_float()
        }

    def total_steps(self):
        return len(self.__dataframe.index)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.__current_row = 0
        self._total_profit = Money(0)

        self.__stock = {
            "symbol": self.__stock["symbol"],
            "qty": Rate(0),
            "avg_price_per_stock": Money(0),
            "type": Position.Short
        }
        try:
            self.__current_date = datetime.strptime(self.__dataframe.iloc[self.__current_row]["Date"],
                                                    "%Y-%m-%d %H:%M:%S")
        except:
            self.__current_date = datetime.strptime(self.__dataframe.iloc[self.__current_row]["Date"],
                                                    "%Y-%m-%d %H:%M")
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.squeeze(action)
        _action = Action(action[0])
        _qty = action[1]
        self.__callback = action[2:]

        # self.__callback = action[2:]

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
                reward += self._close_positions(quantity=_qty)
            else:
                self._buy(quantity=_qty)

        elif _action == Action.Sell:
            if self.__stock["type"] == Position.Long and self.__stock["qty"] != 0:
                reward += self._close_positions(quantity=_qty)
            else:
                self._sell(quantity=_qty)

        self.__current_row += 1
        terminated = self.__current_row == len(self.__dataframe.index) - 1

        if not terminated:

            try:
                new_date = datetime.strptime(self.__dataframe.iloc[self.__current_row]["Date"],
                                             "%Y-%m-%d %H:%M:%S")
            except:
                new_date = datetime.strptime(self.__dataframe.iloc[self.__current_row]["Date"],
                                             "%Y-%m-%d %H:%M")

            if new_date.day != self.__current_date.day:
                reward += self._new_day(new_date)
            self.__current_date = new_date
        else:
            print(
                f"Profit: {self._total_profit}, {(self.__get_current_equity() - self.__starting_equity) / self.__starting_equity * 100}%")
            print(f"Start: {self.__starting_equity}$, end: {self.__get_current_equity()}$")

            reward += (self.__get_current_equity() - self.__starting_equity) / self.__starting_equity

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self, mode='human'):
        pass
