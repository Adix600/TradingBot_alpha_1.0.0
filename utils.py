import sqlite3
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

CONFIG = {
    'symbol': 'EURUSD',
    'lookback_window': 50,
    'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment'],
    'initial_balance': 10000,
    'commission': 0.0002,
    'lstm_hidden': 64,
    'timeframe': mt5.TIMEFRAME_M1,
    'lot_size': 0.1,
    'slippage': 5,
    'simulate': True
}

def get_db_path():
    return "simulated_trades.db" if CONFIG['simulate'] else "live_trades.db"

def init_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
        timestamp TEXT, action TEXT, price REAL, reward REAL, balance REAL)''')
    conn.commit()
    return conn

def log_trade(conn, action, price, reward, balance):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
                   (datetime.now().isoformat(), action, price, reward, balance))
    conn.commit()

def fetch_live_mt5_data(symbol, bars=50):
    if not mt5.initialize():
        raise RuntimeError("[Błąd] Inicjalizacja MT5 nie powiodła się")
    rates = mt5.copy_rates_from_pos(symbol, CONFIG['timeframe'], 0, bars)
    if rates is None or len(rates) == 0:
        raise ValueError("[Błąd] Brak danych z MT5")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']].rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
