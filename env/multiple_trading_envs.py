from random import shuffle

import dask.dataframe as dd
import gymnasium as gym
import pandas_ta as ta

from .defaults import default_strategy
from .trading_env import TradingEnv


class MultipleTradingEnvs(gym.Env):
    def __init__(self, files: list[str], strategy: ta.Strategy = default_strategy):
        self.__partitions = []
        self.__env: TradingEnv
        self._next_env()
        self.__is_first_env = True

        self.__strategy = strategy

        self.action_space = self.__env.action_space
        self.observation_space = self.__env.observation_space

        for file_path in files:
            self.__partitions.append(dd.read_csv(file_path).partitions)

        shuffle(self.__partitions)

    def _next_env(self):
        if self.__is_first_env:
            self.__is_first_env = True
        else:
            self.__env = TradingEnv(self.__partitions.pop().compute(), self.__strategy)

    def reset(self, seed=None, options=None):
        self._next_env()
        return self.__env.reset()

    def step(self, action):
        return self.__env.step(action)


gym.register("MultipleTradingEnvs-V0", MultipleTradingEnvs)
