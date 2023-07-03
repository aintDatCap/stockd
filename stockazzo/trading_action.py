from enum import Enum
from decimal import Decimal
from stockholm import Money
from utils import to_money


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"


class TradingAction:
    def __init__(self, symbol: str, price_per_stock: int | float | str | Decimal | Money,
                 quantity: int | float | str | Decimal | Money,
                 action_type: str | ActionType, leverage: int | float | str | Decimal | Money):

        self.symbol = symbol

        self.price_per_stock = to_money(price_per_stock)

        if self.price_per_stock <= 0:
            raise ValueError(
                f"The price_per_stock should be higher than zero.\nProvided price_per_stock: {self.price_per_stock}")

        self.quantity = to_money(quantity)

        self.action_type = ActionType(action_type)

        self.leverage = to_money(leverage)

        if self.leverage < 1:
            raise ValueError(f"The leverage should be equal or higher than one.\nProvided leverage: {self.leverage}")

    def get_profit_or_loss(self, new_price_per_stock: int | float | str | Decimal | Money) -> Money:
        new_price_per_stock = to_money(new_price_per_stock)

        if self.action_type == ActionType.BUY:
            return (new_price_per_stock - self.price_per_stock) * self.quantity * self.leverage
        else:
            return (self.price_per_stock - new_price_per_stock) * self.quantity * self.leverage

    def get_profit_or_loss_percentage(self, new_price_per_stock: int | float | str | Decimal | Money) -> Money:
        return (self.price_per_stock - new_price_per_stock) * self.leverage / self.price_per_stock

    def to_dict(self):
        return {
            "price_per_stock": str(self.price_per_stock),
            "stock_quantity": str(self.quantity),
            "action_type": self.action_type.value,
            "leverage": str(self.leverage)
        }

    def to_model_array(self) -> list[float]:
        return [
            float(ord(self.symbol)),
            float(self.price_per_stock),
            float(self.quantity),
            1.0 if self.action_type == ActionType.BUY else 0.0,
            float(self.leverage)
        ]