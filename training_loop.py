import pandas as pd
from sb3_contrib import RecurrentPPO
from agent_policy import CustomLSTMPolicy
from env_market import MarketEnv
from mt5_data_loader import load_mt5_data
from utils import CONFIG

def train_until_stagnation(symbol='EURUSD', timeframe='M5', months=6,
                           max_epochs=50, timesteps_per_epoch=5000, stagnation_limit=10):
    df = load_mt5_data(symbol=symbol, timeframe=timeframe, months=months)
    env = MarketEnv(df, window_size=CONFIG['lookback_window'], initial_balance=CONFIG['initial_balance'])

    model = RecurrentPPO(
        policy=CustomLSTMPolicy,
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./ppo_lstm_log/"
    )

    best_balance = -float("inf")
    stagnation_count = 0

    for epoch in range(max_epochs):
        model.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        current_balance = env.balance

        print(f"[Epoch {epoch+1}] Balance: {current_balance:.2f}")

        if current_balance > best_balance:
            best_balance = current_balance
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_limit:
            print(f"[✓] Trening zatrzymany — brak poprawy przez {stagnation_limit} rund.")
            break

    model.save("models/ppo_lstm_model")
    print("[✓] Model PPO+LSTM zapisany.")

if __name__ == "__main__":
    train_until_stagnation()
