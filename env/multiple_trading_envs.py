from random import shuffle

import dask.dataframe as dd
import gymnasium as gym
import pandas_ta as ta

from .defaults import default_strategy
from .trading_env import TradingEnv


class MultipleTradingEnvs(gym.Env):
    def __init__(self, files: list[str], strategy: ta.Strategy = default_strategy):
        self.__partitions: list[dd.DataFrame] = []
        for file_path in files:
            self.__partitions.extend(dd.read_csv(file_path, dtype={'volume': 'float64'})
                                     .repartition(npartitions=30)
                                     .partitions
                                     )

        shuffle(self.__partitions)

        self.__strategy = strategy

        self.__env: TradingEnv
        self._next_env()

        self.action_space = self.__env.action_space
        self.observation_space = self.__env.observation_space

    def _next_env(self):
        self.__env = TradingEnv(self.__partitions.pop().compute(), self.__strategy)

    def get_total_steps(self):
        total = 0
        for partition in self.__partitions:
            total += len(partition.index)
        return total - 180 * len(self.__partitions)

    def reset(self, seed=None, options=None):
        self._next_env()
        return self.__env.reset()

    def step(self, action):
        return self.__env.step(action)
