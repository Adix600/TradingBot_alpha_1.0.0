import optuna
from sb3_contrib import RecurrentPPO
from agent_policy import CustomLSTMPolicy
from env_market import MarketEnv
from mt5_data_loader import load_mt5_data
from utils import CONFIG
import yaml

def objective(trial):
    lookback_window = trial.suggest_int("lookback_window", 30, 100)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    memory_weight = trial.suggest_float("memory_weight", 0.0, 1.0)

    df = load_mt5_data(symbol=CONFIG['symbol'], timeframe="M5", months=6)
    env = MarketEnv(df, window_size=lookback_window, initial_balance=CONFIG['initial_balance'])

    policy_kwargs = {
        "features_extractor_kwargs": {
            "features_dim": 128,
            "hidden_size": hidden_size,
            "dropout": dropout
        }
    }

    model = RecurrentPPO(
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
    return reward_score * (1 + memory_weight)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial)

    # Zapis najlepszych parametrów do config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config.update(study.best_trial.params)

    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print("[✓] Najlepsze parametry zapisane do config.yaml")
