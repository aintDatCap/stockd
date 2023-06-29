import unittest
from decimal import Decimal
from stockazzo import TradingAction


class TestTradingAction(unittest.TestCase):
    def test_profit_or_loss(self):
        action = TradingAction("204.64", 2.6, "sell", 1)

        pl = action.get_profit_or_loss("208.5")
        self.assertAlmostEqual(pl, Decimal("-10.036"))


if __name__ == '__main__':
    unittest.main()
