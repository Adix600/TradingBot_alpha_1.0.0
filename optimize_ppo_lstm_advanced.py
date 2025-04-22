
import optuna
from stable_baselines3 import PPO
from agent_policy import CustomLSTMPolicy
from env_market import MarketEnv
from mt5_data_loader import load_mt5_data
from utils import CONFIG

def objective(trial):
    # Optymalizowane hiperparametry wejściowe i modelowe
    lookback_window = trial.suggest_int("lookback_window", 30, 100)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    memory_weight = trial.suggest_float("memory_weight", 0.0, 1.0)

    # Wczytanie danych MT5 z M5
    df = load_mt5_data(symbol="EURUSD", timeframe="M5", months=6)

    env = MarketEnv(df, window_size=lookback_window, initial_balance=CONFIG['initial_balance'])

    policy_kwargs = {
        "features_extractor_kwargs": {
            "features_dim": 128,
            "hidden_size": hidden_size,
            "dropout": dropout
        }
    }

    model = PPO(
        policy=CustomLSTMPolicy,
        env=env,
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        gamma=trial.suggest_float("gamma", 0.90, 0.999),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.0),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.3),
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    model.learn(total_timesteps=20000)

    final_balance = env.balance if hasattr(env, 'balance') else CONFIG['initial_balance']
    reward_score = final_balance - CONFIG['initial_balance']

    # Nagroda końcowa może uwzględniać wpływ pamięci
    return reward_score * (1 + memory_weight)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial)

    study.trials_dataframe().to_csv("ppo_lstm_advanced_trials.csv")
