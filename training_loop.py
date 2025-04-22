import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from agent_policy import CustomLSTMPolicy
from env_market import MarketEnv
from mt5_data_loader import load_mt5_data
import MetaTrader5 as mt5
from utils import CONFIG
from stable_baselines3.common.logger import configure


def train_until_stagnation(symbol=None, timeframe=None, months=6,
    max_epochs=50, timesteps_per_epoch=5000, stagnation_limit=10):
    symbol = symbol or CONFIG['symbol']
    symbol = symbol or CONFIG['symbol']
    timeframe = mt5.TIMEFRAME_M5
    print(timeframe)
    tf_raw = timeframe or CONFIG['timeframe']
    if isinstance(tf_raw, str):
        timeframe = getattr(mt5, f"TIMEFRAME_{tf_raw}")
    else:
        timeframe = tf_raw
    lookback_window = CONFIG['lookback_window']

    df = load_mt5_data(symbol=symbol, timeframe=timeframe, months=months)
    env = MarketEnv(df, window_size=lookback_window, initial_balance=CONFIG['initial_balance'])

    policy_kwargs = {
        "features_extractor_kwargs": {
            "features_dim": 128,
            "hidden_size": CONFIG['hidden_size'],
            "dropout": CONFIG['dropout']
        }
    }

    model = RecurrentPPO(
        policy=CustomLSTMPolicy,
        env=env,
        learning_rate=CONFIG['lr'],
        n_steps=CONFIG['n_steps'],
        batch_size=CONFIG['batch_size'],
        gamma=CONFIG['gamma'],
        gae_lambda=CONFIG['gae_lambda'],
        clip_range=CONFIG['clip_range'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_lstm_log/"
    )

    # TensorBoard + Pandas logging
    new_logger = configure("./ppo_lstm_log/", ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    best_balance = -float("inf")
    stagnation_count = 0
    equity_curve = []
    initial_balance = CONFIG['initial_balance']

    for epoch in range(max_epochs):
        model.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        current_balance = env.balance
        equity_curve.append(current_balance)

        # Log to TensorBoard
        model.logger.record("train/balance", current_balance)
        model.logger.dump(step=epoch)

        print(f"[Epoch {epoch+1}] Balance: {current_balance:.2f}")

        # Capital management: dynamic position sizing + max drawdown
        drawdown = (initial_balance - current_balance) / initial_balance
        if drawdown >= 0.2:
            print("[❌] Max drawdown przekroczony. Przerywanie treningu.")
            break

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

    # Save equity curve to CSV
    pd.DataFrame({"equity": equity_curve}).to_csv("logs/equity_curve.csv", index=False)
    print("[✓] Equity curve zapisana do logs/equity_curve.csv")


if __name__ == "__main__":
    train_until_stagnation()