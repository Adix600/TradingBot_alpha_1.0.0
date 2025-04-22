import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils import CONFIG

class MarketEnv(gym.Env):
    def __init__(self, df, window_size=50, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(5)  # Hold, Buy, Sell, Close Long, Close Short
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(window_size, len(CONFIG['features'])),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.index = self.window_size
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.index - self.window_size:self.index]
        normed = (window - window.min()) / (window.max() - window.min() + 1e-8)
        return normed.values.astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.index]['Close']
        reward = 0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 3 and self.position == 1:
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0
        elif action == 4 and self.position == -1:
            reward = self.entry_price - price
            self.balance += reward
            self.position = 0

        self.index += 1
        terminated = self.index >= len(self.df) - 1
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {"balance": self.balance}
