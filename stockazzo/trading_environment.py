from decimal import Decimal
from stockholm import Money
from utils import to_money
from trading_action import TradingAction
import gymnasium as gym
import pandas as pd
import os


class TradingEnvironment(gym.Env):
    def __init__(self, starting_equity: int | float | str | Decimal | Money, dataset: pd.DataFrame):
        self.__starting_equity = to_money(starting_equity)

        if self.__starting_equity < 0:
            raise ValueError(
                f"The starting_equity should be equal or higher than 0.\nProvided starting_equity: {starting_equity}")

        self.__current_equity = self.__starting_equity

        self.__actions: list[TradingAction] = []

    def __get_batches(self):
        batches = {}
        for root, _, files in os.walk("/content/elaborated_data"):
            for file_name in files:
                if file_name.endswith(".csv"):
                    symbol = os.path.basename(root)
                    batch_path = os.path.join(root, file_name)

                    if not symbol in batches:
                        batches[symbol] = [batch_path]
                    else:
                        batches[symbol].append(batch_path)
        return batches

    def _get_obs(self):
        return {"actions": self.__actions}

    def reset(self, seed=None, options=None):
        self.__actions.clear()
        self.__current_equity = self.__starting_equity

    def step(self, action):
        if action["action"] == "buy":
            pass

    def add_balance(self, balance: int | float | str | Decimal | Money):
        self.__current_equity += to_money(balance)

    def execute_action(self, action: TradingAction):
        price = action.price_per_stock * action.quantity
        if price / self.__current_equity > 0.05:
            action.quantity = self.__current_equity * 0.05 / action.price_per_stock

        self.__actions.append(action)

    def close_action(self, index: int, closing_price_per_stock):
        action = self.__actions.pop(index)
        self.add_balance(action.get_profit_or_loss(closing_price_per_stock))

    def get_current_equity(self):
        return self.__current_equity

    def get_action_history(self):
        return self.__actions

    @staticmethod
    def __parse_outputs(outputs):
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
