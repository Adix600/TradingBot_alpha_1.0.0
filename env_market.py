import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from utils import CONFIG

# Ustawienia nagrody domyślne (jeśli brak w config.yaml)
CONFIG.setdefault("transaction_cost", 0.0001)
CONFIG.setdefault("quick_exit_penalty", 0.005)
CONFIG.setdefault("idle_penalty", 0.001)
CONFIG.setdefault("unrealized_weight", 0.05)

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
        self.prev_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.index = self.window_size
        self.idle_steps = 0
        self.position_duration = 0
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.index - self.window_size:self.index]
        normed = (window - window.min()) / (window.max() - window.min() + 1e-8)
        return normed.values.astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.index]['Close']
        reward = 0.0
        transaction_cost = CONFIG["transaction_cost"]
        quick_exit_penalty = CONFIG["quick_exit_penalty"]
        idle_penalty = CONFIG["idle_penalty"]
        shaping_factor = CONFIG["unrealized_weight"]

        # Realizacja pozycji
        if action == 3 and self.position == 1:
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0
            if self.position_duration < 3:
                reward -= quick_exit_penalty
        elif action == 4 and self.position == -1:
            reward = self.entry_price - price
            self.balance += reward
            self.position = 0
            if self.position_duration < 3:
                reward -= quick_exit_penalty

        # Otwarcie pozycji
        elif action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            self.position_duration = 0
            reward -= transaction_cost
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price
            self.position_duration = 0
            reward -= transaction_cost

        # Niezrealizowany zysk/strata
        if self.position != 0:
            unrealized = (price - self.entry_price) * self.position
            reward += shaping_factor * unrealized
            self.position_duration += 1

        # Kara za pasywność bez pozycji
        if action == 0 and self.position == 0:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if self.position == 0 and self.idle_steps > 0:
            reward -= idle_penalty * self.idle_steps

        # Kara za stratę
        delta_balance = self.balance - self.prev_balance
        if delta_balance < 0:
            reward += 1.5 * delta_balance  # wzmocniona kara za stratę

        self.prev_balance = self.balance
        self.index += 1
        terminated = self.index >= len(self.df) - 1
        truncated = False
        return self._get_observation(), reward, terminated, truncated, {"balance": self.balance}
