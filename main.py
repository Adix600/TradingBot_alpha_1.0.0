import random
import os
from datetime import datetime, timedelta
from config import forex_symbols
from agent import RLAgent, train_agent
from simulator import backtest, run_live
from data_fetcher import get_data, fetch_sentiment
from features import prepare_data, get_state
import MetaTrader5 as mt5
import torch


def main():
    symbol = random.choice(forex_symbols)
    print(f"🎯 Wybrany symbol: {symbol}")
    end = datetime.now() - timedelta(days=30)
    start = end - timedelta(days=365 * 20)
    base = start + timedelta(days=random.randint(0, (end - start).days - 1825))
    df = get_data(symbol, base, base + timedelta(days=365))
    sentiment = fetch_sentiment(symbol, base.strftime('%Y-%m-%d'), (base + timedelta(days=365)).strftime('%Y-%m-%d'))
    df = prepare_data(df, sentiment, symbol)
    sample_state = get_state(df.iloc[0])
    agent = train_agent(df[:int(0.8 * len(df))], input_size=len(sample_state))
    backtest(agent, df[int(0.8 * len(df)):])

if __name__ == '__main__':
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5")
        exit(1)

    try:
        print("\n[MENU]")
        print("1. Trening i Backtest")
        print("2. Tryb LIVE")
        print("3. Tylko Backtest")
        choice = input("Wybierz (1/2/3): ").strip()

        if choice == '1':
            main()

        elif choice == '2':
            agent = RLAgent(input_size=9)
            if os.path.exists("trained_agent.pth"):
                agent.load_state_dict(torch.load("trained_agent.pth"))
                run_live(agent, 'EURUSD', mt5)
            else:
                print("Brak wytrenowanego modelu. Najpierw uruchom trening.")

        elif choice == '3':
            agent = RLAgent(input_size=9)
            if os.path.exists("trained_agent.pth"):
                agent.load_state_dict(torch.load("trained_agent.pth"))
                symbol = random.choice(forex_symbols)
                print(f"🔍 Backtest na symbolu: {symbol}")
                end = datetime.now() - timedelta(days=30)
                start = end - timedelta(days=365 * 20)
                base = start + timedelta(days=random.randint(0, (end - start).days - 1825))
                df = get_data(symbol, base, base + timedelta(days=365))
                sentiment = fetch_sentiment(symbol, base.strftime('%Y-%m-%d'), (base + timedelta(days=365)).strftime('%Y-%m-%d'))
                df = prepare_data(df, sentiment, symbol)
                backtest(agent, df[int(0.8 * len(df)):])
            else:
                print("Brak wytrenowanego modelu.")

        else:
            print("Nieprawidłowy wybór.")
    finally:
        mt5.shutdown()
