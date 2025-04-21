import gym
import numpy as np
import pandas as pd
from gym import spaces
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class HistoricalTradingEnv(gym.Env):
    def __init__(self, symbol='EURUSD', timeframe=mt5.TIMEFRAME_M1, window_size=50, months=6):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.months = months
        self.data = self.load_data()
        self.max_steps = len(self.data) - window_size - 1
        self.step_index = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, 5), dtype=np.float32)

        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000

    def load_data(self):
        if not mt5.initialize():
            raise RuntimeError("MetaTrader5 initialization failed")
        end = datetime.now()
        start = end - timedelta(days=self.months * 30)
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, start, end)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def reset(self):
        self.step_index = 0
        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000
        return self._get_observation()

    def _get_observation(self):
        window = self.data.iloc[self.step_index:self.step_index + self.window_size].copy()
        window = (window - window.min()) / (window.max() - window.min() + 1e-8)
        return window.values.astype(np.float32)

    def step(self, action):
        current_price = self.data.iloc[self.step_index + self.window_size]['Close']
        reward = 0
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
        elif action == 3 and self.position == 1:
            reward = current_price - self.entry_price
            self.balance += reward
            self.position = 0
        elif action == 4 and self.position == -1:
            reward = self.entry_price - current_price
            self.balance += reward
            self.position = 0
        self.step_index += 1
        done = self.step_index + self.window_size >= len(self.data)
        return self._get_observation(), reward, done, {'balance': self.balance}
