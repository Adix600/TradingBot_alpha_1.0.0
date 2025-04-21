# runner.py
import time
import os
import numpy as np
from datetime import datetime
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from utils import CONFIG, init_db, get_db_path, fetch_live_mt5_data
from sentiment import scrape_latest_news, analyze_sentiment
from trading import simulate_trade, sim_state
from reporting import export_html_report
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_live_bot(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Błąd] Nie znaleziono modelu PPO: {model_path}")

    sentiment_pipe = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    # jeśli chciałbyś wektorowe envy:
    # env = DummyVecEnv([lambda: LiveTradingEnv()])
    model = PPO.load(model_path, device=device)

    db_path = get_db_path()
    db_conn = init_db(db_path)

    print("[Start] Uruchomiono bota. Tryb:", "Symulacja" if CONFIG['simulate'] else "Live")

    while True:
        try:
            df = fetch_live_mt5_data(CONFIG['symbol'], CONFIG['lookback_window'])
            headlines = scrape_latest_news()
            sentiment = np.mean(analyze_sentiment(headlines, sentiment_pipe))
            df['Sentiment'] = sentiment
            df.fillna(method='ffill', inplace=True)
            obs = np.expand_dims(MinMaxScaler().fit_transform(df.values), axis=0).astype(np.float32)

            action, _ = model.predict(obs)
            price = df.iloc[-1]['Close']
            action_name = ['Hold','Buy','Sell','Close Long','Close Short'][action]
            print(f"[{datetime.now()}] Action: {action_name}")

            if CONFIG['simulate']:
                reward = simulate_trade(action, price, db_conn)
                print(f"Simulated Balance: {sim_state['balance']:.2f}, Reward: {reward:.5f}")

            if datetime.now().minute % 30 == 0:
                print("[Info] Rozpoczynam trening PPO...")
                model.learn(total_timesteps=1000, reset_num_timesteps=False)
                model.save(model_path)
                export_html_report(db_path)
                print("[Info] Trening zakończone i zapisane. Raport zaktualizowany.")

            time.sleep(60)
        except Exception as e:
            print(f"[Błąd] {e}")
            time.sleep(60)

if __name__ == "__main__":
    # jeśli chcesz wywołać z main.py, usuń to wywołanie
    run_live_bot("models/ppo_model.zip")
