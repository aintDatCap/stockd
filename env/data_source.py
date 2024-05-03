from abc import ABC, abstractmethod
import pandas as pd


class TradingDataSource(ABC):
    @abstractmethod
    def next_data_batch(self) -> pd.DataFrame | None:
        return None
