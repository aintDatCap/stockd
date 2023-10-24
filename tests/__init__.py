import pandas as pd
from pathlib import Path
from env import TradingEnv, default_strategy
from stable_baselines3 import PPO


if __name__ == "__main__":
    relative_path = Path(__file__).parent
    df = pd.read_csv(relative_path / "data/eurusd_sample.csv")
    env = TradingEnv(df, default_strategy)

    model = PPO('MultiInputPolicy', env, verbose=0)
    model.learn(20_000)
