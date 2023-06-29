from decimal import Decimal
from stockholm import Money
from utils import to_money
from trading_action import TradingAction


class TradingEnvironment:
    def __init__(self, starting_equity: int | float | str | Decimal | Money):
        self.__starting_equity = to_money(starting_equity)

        if self.__starting_equity < 0:
            raise ValueError(
                f"The starting_equity should be equal or higher than 0.\nProvided starting_equity: {starting_equity}")

        self.__current_equity = self.__starting_equity

        self.__actions: list[TradingAction] = []

        self.__assets: dict = {}

    def add_balance(self, balance: int | float | str | Decimal | Money):
        self.__current_equity += to_money(balance)

    def execute_action(self, action: TradingAction):
        self.__actions.append(action)

    def get_current_equity(self):
        return self.__current_equity

    def get_action_history(self):
        return self.__actions
