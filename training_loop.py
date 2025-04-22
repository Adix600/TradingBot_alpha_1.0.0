import pandas as pd
from sb3_contrib import RecurrentPPO
from agent_policy import CustomLSTMPolicy
from env_market import MarketEnv
from mt5_data_loader import load_mt5_data
from utils import CONFIG

def train_until_stagnation(symbol=None, timeframe=None, months=6,
                           max_epochs=50, timesteps_per_epoch=5000, stagnation_limit=10):
    symbol = symbol or CONFIG['symbol']
    timeframe = timeframe or CONFIG['timeframe']
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
