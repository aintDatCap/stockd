import gymnasium as gym
import dask.dataframe as dd
from random import shuffle
from .trading_env import TradingEnv


class MultipleTradingEnvs(gym.Env):
    def __init__(self, files: list[str]):
        self.__partitions = []
        self.__env: TradingEnv
        for file_path in files:
            self.__partitions.append(dd.read_csv(file_path).partitions)

        shuffle(self.__partitions)

    def _next_env(self):
        self.__env = TradingEnv(self.__partitions.pop().compute())

    def reset(self, seed=None, options=None):
        self._next_env()
        return self.__env.reset()

    def step(self, action):
        return self.__env.step(action)


gym.register("MultipleTradingEnvs-V0", MultipleTradingEnvs)
