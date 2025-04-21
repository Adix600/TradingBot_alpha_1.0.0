# utils.py
import sqlite3
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import yaml

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # zamień timeframe string na stałą MT5
    if isinstance(cfg.get('timeframe'), str):
        cfg['timeframe'] = getattr(mt5, 'TIMEFRAME_' + cfg['timeframe'])
    return cfg

CONFIG = load_config()

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

def fetch_spread(symbol):
    """Zwraca bieżący spread (ask – bid) dla danego symbolu."""
    if not mt5.initialize():
        raise RuntimeError(f"[Błąd] Inicjalizacja MT5 nie powiodła się przy pobieraniu spreadu dla {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"[Błąd] Nie udało się pobrać tików dla {symbol}")
    return float(tick.ask - tick.bid)