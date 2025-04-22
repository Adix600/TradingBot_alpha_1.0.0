import time
import os
import numpy as np
from datetime import datetime
from utils import CONFIG, init_db, get_db_path
from sentiment import scrape_latest_news, analyze_sentiment
from trend_infer import predict_trend_score
from replay_lookup import load_replay_memory, find_similar_case, save_replay_memory
from sb3_contrib import RecurrentPPO
from agent_policy import load_lstm_trend_model
import subprocess
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mt5_data_loader import load_mt5_data

def run_live_bot(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Błąd] Nie znaleziono modelu PPO: {model_path}")

    print("[Start] Uruchamianie bota...")
    ppo_model = RecurrentPPO.load(model_path)
    lstm_model = load_lstm_trend_model()
    db_path = get_db_path()
    db_conn = init_db(db_path)
    memory = load_replay_memory()

    while True:
        try:
            df = load_mt5_data(symbol=CONFIG['symbol'], timeframe=CONFIG['timeframe'], months=1)
            headlines = scrape_latest_news()
            sentiment = np.mean(analyze_sentiment(headlines, pipe=None))
            df['Sentiment'] = sentiment
            trend_score = predict_trend_score(df.copy(), lstm_model)
            df['Trend'] = trend_score
            df = df[CONFIG['features']]
            df.fillna(method='ffill', inplace=True)

            scaler = MinMaxScaler()
            obs = scaler.fit_transform(df.tail(CONFIG['lookback_window']).values)
            obs = np.expand_dims(obs, axis=0).astype(np.float32)

            if CONFIG.get("use_memory", False):
                is_similar, case = find_similar_case(obs, memory)
                if is_similar and case["reward"] < 0:
                    print("[⚠️ Replay] Ostrzeżenie: podobna sytuacja zakończyła się stratą.")

            action, _ = ppo_model.predict(obs, deterministic=True)
            action_name = ['Hold', 'Buy', 'Sell', 'Close Long', 'Close Short'][int(action)]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Decyzja RecurrentPPO: {action_name}")

            reward = np.random.normal(0, 0.01)  # symulowany reward

            if int(action) in [3, 4]:
                memory.append({
                    "state": obs,
                    "action": int(action),
                    "reward": float(reward),
                    "time": time.time()
                })
                save_replay_memory(memory)
                print("[🧠] Zapisano przypadek do replay memory.")

            if datetime.now().weekday() == 6 and datetime.now().hour == 6:
                print("[🧹] Czyszczenie replay memory...")
                subprocess.run(["python", "clean_replay_memory.py"])

            time.sleep(60)
        except Exception as e:
            print(f"[Błąd] {e}")
            time.sleep(60)
