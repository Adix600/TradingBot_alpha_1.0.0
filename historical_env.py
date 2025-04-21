# historical_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from utils import fetch_spread, CONFIG

class HistoricalTradingEnv(gym.Env):
    def __init__(self, symbol='EURUSD', timeframe=None, window_size=50, months=6):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe or CONFIG['timeframe']
        self.window_size = window_size
        self.months = months
        self.data = self.load_data()
        self.max_steps = len(self.data) - window_size - 1
        self.step_index = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(window_size, 5), dtype=np.float32
        )

        self.position = 0
        self.entry_price = 0.0
        self.balance = CONFIG['initial_balance']
        self.peak_balance = self.balance

        # stały koszt transakcji
        self.spread = fetch_spread(self.symbol)

    def load_data(self):
        if not mt5.initialize():
            raise RuntimeError("MetaTrader5 init failed")
        end = datetime.now()
        start = end - timedelta(days=self.months * 30)
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, start, end)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        return df[['Open','High','Low','Close','Volume']]

    def reset(self):
        self.step_index = 0
        self.position = 0
        self.entry_price = 0.0
        self.balance = CONFIG['initial_balance']
        self.peak_balance = self.balance
        return self._get_observation()

    def _get_observation(self):
        window = self.data.iloc[
            self.step_index : self.step_index + self.window_size
        ].copy()
        window = (window - window.min()) / (window.max() - window.min() + 1e-8)
        return window.values.astype(np.float32)

    def step(self, action):
        current_price = self.data.iloc[
            self.step_index + self.window_size
        ]['Close']
        reward = 0
        risk_amount = self.peak_balance * CONFIG['risk_percent']

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
        elif action == 3 and self.position == 1:
            pnl = current_price - self.entry_price - self.spread
            reward = pnl * (risk_amount / CONFIG['lot_size'])
            self.balance += reward
            self.position = 0
        elif action == 4 and self.position == -1:
            pnl = self.entry_price - current_price - self.spread
            reward = pnl * (risk_amount / CONFIG['lot_size'])
            self.balance += reward
            self.position = 0

        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        done = drawdown > CONFIG['max_drawdown'] or (
            self.step_index + self.window_size >= len(self.data)
        )

        self.step_index += 1
        return self._get_observation(), reward, done, {
            'balance': self.balance,
            'drawdown': drawdown
        }
