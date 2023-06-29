from enum import Enum
from decimal import Decimal
from fractions import Fraction


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"


class TradingAction:
    def __init__(self, price_per_stock: int | float | str | Decimal, possessed_quantity: int | float | str | Decimal,
                 action_type: str | ActionType, leverage: int | float | str | Decimal):

        self.price_per_stock = self.__to_decimal(price_per_stock)

        if self.price_per_stock <= 0:
            raise ValueError(
                f"The price_per_stock should be higher than zero.\nProvided price_per_stock: {self.price_per_stock}")

        self.possessed_quantity = self.__to_decimal(possessed_quantity)

        self.action_type = ActionType(action_type)

        self.leverage = self.__to_decimal(leverage)

        if self.leverage < 1:
            raise ValueError(f"The leverage should be equal or higher than one.\nProvided leverage: {self.leverage}")

    @staticmethod
    def __to_decimal(value) -> Decimal:
        if not isinstance(value, Decimal):
            return Decimal(value)
        return value

    def get_profit_or_loss(self, new_price_per_stock: int | float | str | Decimal) -> Decimal:
        new_price_per_stock = self.__to_decimal(new_price_per_stock)

        if self.action_type == ActionType.BUY:
            return (new_price_per_stock - self.price_per_stock) * self.possessed_quantity * self.leverage
        else:
            return (self.price_per_stock - new_price_per_stock) * self.possessed_quantity * self.leverage

    def get_profit_or_loss_percentage(self, new_price_per_stock: int | float | str | Decimal) -> Decimal:
        pl = self.get_profit_or_loss(new_price_per_stock)

        return pl / self.price_per_stock

    def to_dict(self):
        return {
            "price_per_stock": str(self.price_per_stock),
            "stock_quantity": str(self.possessed_quantity),
            "action_type": self.action_type.value,
            "leverage": str(self.leverage)
        }
