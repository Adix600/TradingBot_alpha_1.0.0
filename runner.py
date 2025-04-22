import time
import os
import numpy as np
from datetime import datetime
from utils import CONFIG, init_db, get_db_path, log_trade
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
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading


def plot_equity_curve(equity_curve, filename="logs/live_equity_curve.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve, label="Equity")
    plt.title("Krzywa Equity - Live Trading")
    plt.xlabel("Krok")
    plt.ylabel("Saldo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def show_live_dashboard():
    def update_plot():
        while True:
            try:
                df = pd.read_csv("logs/live_equity_curve.csv")
                ax.clear()
                ax.plot(df['equity'], label="Equity")
                ax.set_title("Live Equity Tracker")
                ax.set_xlabel("Tick")
                ax.set_ylabel("Balance")
                ax.legend()
                ax.grid(True)
                fig.tight_layout()
                canvas.draw()
            except Exception as e:
                print(f"[LivePlot] Błąd: {e}")
            time.sleep(5)

    root = tk.Tk()
    root.title("Live Equity Dashboard")
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    threading.Thread(target=update_plot, daemon=True).start()
    threading.Thread(target=root.mainloop, daemon=True).start()


def run_live_bot(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Błąd] Nie znaleziono modelu PPO: {model_path}")

    print("[Start] Uruchamianie bota...")
    ppo_model = RecurrentPPO.load(model_path)
    lstm_model = load_lstm_trend_model()
    db_path = get_db_path()
    db_conn = init_db(db_path)
    memory = load_replay_memory()

    equity_curve = []
    initial_balance = CONFIG['initial_balance']
    balance = initial_balance

    show_live_dashboard()  # Uruchomienie dashboardu razem z botem

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
                    print("[\u26a0\ufe0f Replay] Ostrzeżenie: podobna sytuacja zakończyła się stratą.")

            action, _ = ppo_model.predict(obs, deterministic=True)
            action_name = ['Hold', 'Buy', 'Sell', 'Close Long', 'Close Short'][int(action)]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Decyzja RecurrentPPO: {action_name}")

            reward = np.random.normal(0, 0.01)  # symulowany reward
            balance += reward
            equity_curve.append(balance)

            log_trade(db_conn, action_name, df.iloc[-1]['Close'], reward, balance)

            # Max drawdown check
            drawdown = (initial_balance - balance) / initial_balance
            if drawdown >= 0.2:
                print("[\u274c] Max drawdown przekroczony. Zatrzymywanie bota.")
                break

            if int(action) in [3, 4]:
                memory.append({
                    "state": obs,
                    "action": int(action),
                    "reward": float(reward),
                    "time": time.time()
                })
                save_replay_memory(memory)
                print("[\ud83e\udde0] Zapisano przypadek do replay memory.")

            if datetime.now().weekday() == 6 and datetime.now().hour == 6:
                print("[\ud83e\uddf9] Czyszczenie replay memory...")
                subprocess.run(["python", "clean_replay_memory.py"])

            pd.DataFrame({"equity": equity_curve}).to_csv("logs/live_equity_curve.csv", index=False)
            plot_equity_curve(equity_curve)

            time.sleep(60)
        except Exception as e:
            print(f"[Błąd] {e}")
            time.sleep(60)


if __name__ == "__main__":
    print("[Info] Użyj pliku main.py, aby uruchomić bota i dashboard z menu CLI.")
