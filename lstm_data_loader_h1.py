import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_lstm_data_from_mt5(symbol='EURUSD', timeframe='H1', months=12, window=60):
    if not mt5.initialize():
        raise RuntimeError("[MT5] Inicjalizacja nie powiodła się")

    tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
    end = datetime.now()
    start = end - timedelta(days=months * 30)

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        raise ValueError("[MT5] Brak danych")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)

    df['Sentiment'] = 0.0
    df['Trend'] = 0.0
    df['Sentiment'] = df['Sentiment'].rolling(3, min_periods=1).mean()
    df['Trend'] = df['Trend'].rolling(3, min_periods=1).mean()

    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Trend']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window][3])  # wartość Close jako target

    return np.array(X), np.array(y), scaler

def prepare_lstm_sequences(df, window=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X = []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
    return np.array(X), None, scaler
