import sqlite3
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import yaml

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg.get('timeframe'), str):
        cfg['timeframe'] = getattr(mt5, 'TIMEFRAME_' + cfg['timeframe'])
    return cfg

CONFIG = load_config()
CONFIG.setdefault("use_memory", True)

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

def fetch_spread(symbol):
    if not mt5.initialize():
        raise RuntimeError(f"[Błąd] MT5: brak połączenia z terminalem")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"[Błąd] Nie udało się pobrać danych tickowych dla {symbol}")
    return float(tick.ask - tick.bid)
