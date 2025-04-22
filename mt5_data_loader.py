import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def load_mt5_data(symbol="EURUSD", timeframe="M5", months=12) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError("[MT5] Inicjalizacja nie powiodła się")

    tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
    end = datetime.now()
    start = end - timedelta(days=months * 30)

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        raise ValueError("[MT5] Brak danych historycznych")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)

    df['Sentiment'] = 0.0
    df['Trend'] = 0.0

    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Trend']]
    return df
