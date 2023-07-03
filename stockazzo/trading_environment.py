import os.path
from decimal import Decimal
from random import randrange

import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.envs.registration import register
from stockholm import Money

from trading_action import TradingAction
from utils import to_money


class TradingEnvironment(gym.Env):
    def __init__(self, starting_equity: int | float | str | Decimal | Money, dataset_paths: list[str]):
        self.__starting_equity = to_money(starting_equity)

        if self.__starting_equity < 0:
            raise ValueError(
                f"The starting_equity should be equal or higher than 0.\nProvided starting_equity: {starting_equity}")

        self.__current_equity = self.__starting_equity

        self.__actions: list[TradingAction] = []

        self.__dataset_paths = dataset_paths

        self.__current_dataset: pd.DataFrame

        self.__row_index = 0

        self.__rewards: list[float] = []

        self.__callback = np.zeros(3)

        self.__current_dataset_path = ""

    def __next_dataset(self):
        self.__current_dataset_path = self.__dataset_paths.pop(randrange(len(self.__dataset_paths)))
        self.__current_dataset = pd.read_csv(self.__current_dataset_path)

    def _get_obs(self):
        return {"actions": self.__actions}

    def _get_info(self):
        return {"p/l": self.get_profit_loss(), "p/l%": self.get_profit_loss_percentage()}

    def get_inputs(self):
        stock_inputs = np.zeros((5, 5))
        for i, action in enumerate(self.__actions):
            stock_inputs[i] = np.array(action.to_model_array())

        current_inputs = np.zeros((1, 5))  # latest price, current equity and callback

        current_inputs[0][0] = float(self.__current_dataset[self.__row_index]["closed"])
        current_inputs[0][1] = self.__current_equity.as_float()

        for i in range(2, 5):
            current_inputs[0][i] = self.__callback[i - 1]

        inputs = np.concatenate([stock_inputs, current_inputs])
        return np.array([inputs])

    def reset(self, seed=None, options=None):
        self.__actions.clear()
        self.__rewards.clear()
        self.__current_equity = self.__starting_equity
        self.__next_dataset()

        return self._get_obs(), self._get_info()

    def step(self, action, symbol=None):
        data = self.__current_dataset[self.__row_index]

        reward = 0
        terminated = self.__row_index == len(self.__current_dataset) - 1

        if symbol is None:
            symbol = os.path.basename(self.__current_dataset_path)

        if action["action"] == "buy":
            self.execute_action(TradingAction(symbol,
                                              data["closed"],
                                              action["quantity"],
                                              action["action"],
                                              action["leverage"]
                                              ))

        elif action["action"] == "sell":
            max_index = 0

            for i, _action in enumerate(self.__actions):
                if _action.get_profit_or_loss(data["closed"]) > self.__actions[max_index].get_profit_or_loss(
                        data["closed"]):
                    max_index = i

            profit_loss_percentage = self.close_action(max_index, data["closed"])

            reward = pow(profit_loss_percentage * 1000, 3)
            self.__rewards.append(reward)

        self.__row_index += 1

        return self._get_obs(), reward, terminated, False, self._get_info()

    def close(self):
        pass

    def add_balance(self, balance: int | float | str | Decimal | Money):
        self.__current_equity += to_money(balance)

    def execute_action(self, action: TradingAction):
        price = action.price_per_stock * action.quantity
        if price / self.__current_equity > 0.05:
            action.quantity = self.__current_equity * 0.05 / action.price_per_stock

        if self.__starting_equity < 2000 and action.leverage > 1:
            action.leverage = 1

        self.__actions.append(action)

    def close_action(self, index: int, closing_price_per_stock):
        action = self.__actions.pop(index)
        self.add_balance(action.get_profit_or_loss(closing_price_per_stock))
        return action.get_profit_or_loss_percentage(closing_price_per_stock)

    def get_current_equity(self):
        return self.__current_equity

    def get_action_history(self):
        return self.__actions

    def get_profit_loss(self) -> Money:
        return self.__current_equity - self.__starting_equity

    def get_profit_loss_percentage(self):
        return self.get_profit_loss() / self.__starting_equity

    def set_callback(self, callback: list[float]):
        if len(callback) != 3:
            raise ValueError("There should only 3 callback inputs")
        self.__callback = callback

    @staticmethod
    def parse_outputs(outputs):
        result = {}
        outputs = outputs.numpy()[0][0]

        outputs[1] = abs(outputs[1])
        outputs[2] = abs(outputs[2])

        for i in range(3):
            outputs[i] = outputs[i] / 1000

        result["action"] = "buy" if outputs[0] >= 1 else "sell" if outputs[0] <= -1 else "stay"
        result["quantity"] = 0.05 if outputs[1] >= 0.05 else outputs[1]
        result["leverage"] = 3 if outputs[2] >= 3 else 1 if outputs[2] <= 1 else outputs[2]
        result["callback"] = outputs[3:len(outputs)]

        return result


register(
    id="aintDatCap/TradingEnv-0",
    entry_point="TradingEnvironment",
)
