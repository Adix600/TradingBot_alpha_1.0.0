import time
import numpy as np
import pandas as pd
from mt5_data_loader import load_mt5_data
from replay_lookup import save_replay_memory
from sklearn.preprocessing import MinMaxScaler

def generate_random_action():
    return np.random.choice([1, 2, 3, 4])  # pomijamy Hold (0)

def simulate_reward(position, entry_price, exit_price):
    if position == 1:
        return exit_price - entry_price
    elif position == -1:
        return entry_price - exit_price
    return 0.0

def preload_replay_from_history(symbol='EURUSD', timeframe='M5', months=6, window=50):
    df = load_mt5_data(symbol=symbol, timeframe=timeframe, months=months)
    df.fillna(method='ffill', inplace=True)
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    memory = []
    position = 0
    entry_price = 0.0

    for i in range(window, len(df) - 1):
        obs = df.iloc[i - window:i].values
        next_price = df.iloc[i + 1]['Close']
        action = generate_random_action()
        reward = 0.0

        if action in [1, 2] and position == 0:
            position = 1 if action == 1 else -1
            entry_price = df.iloc[i]['Close']
        elif action == 3 and position == 1:
            reward = simulate_reward(position, entry_price, next_price)
            position = 0
        elif action == 4 and position == -1:
            reward = simulate_reward(position, entry_price, next_price)
            position = 0

        memory.append({
            "state": np.expand_dims(obs, axis=0),
            "action": int(action),
            "reward": float(reward),
            "time": time.time()
        })

    save_replay_memory(memory)
    print(f"[✓] Wygenerowano {len(memory)} wspomnień na podstawie danych historycznych.")

if __name__ == "__main__":
    preload_replay_from_history()
