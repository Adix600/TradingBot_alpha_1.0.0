import MetaTrader5 as mt5
import pandas as pd
import numpy as np


def get_ema200(symbol):
    """
    Function fetches data from MetaTrader5 and calculates EMA200 for the given symbol.
    Returns the EMA200 value on the last candle.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)  # Get last 200 candles
    if rates is None or len(rates) == 0:
        raise ValueError(f"Failed to fetch data for {symbol} from MetaTrader 5.")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema200'] = df['close'].ewm(span=200).mean()  # Calculate EMA200
    return df['ema200'].iloc[-1]  # Return EMA200 on last candle


def prepare_data(df, sentiment, symbol):
    """
    Function prepares data for the model, including technical indicators and sentiment.
    """
    df = df.copy()

    # Technical indicators
    df['ma_short'] = df['close'].rolling(5).mean()
    df['ma_long'] = df['close'].rolling(15).mean()

    # Correct RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['volatility'] = df['close'].rolling(10).std()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['stoch'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / (
                df['high'].rolling(14).max() - df['low'].rolling(14).min()))

    # Add EMA200
    df['ema200'] = get_ema200(symbol)

    df = df.dropna()

    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = ['ma_short', 'ma_long', 'rsi', 'volatility', 'momentum', 'macd', 'stoch', 'ema200']
    df[features] = scaler.fit_transform(df[features])

    # Add sentiment
    df['sentiment'] = sentiment

    return df


def get_state(row):
    """
    Function transforms one row of data into a feature vector.
    """
    return np.array([
        row['ma_short'], row['ma_long'], row['rsi'], row['volatility'],
        row['momentum'], row['macd'], row['stoch'], row['sentiment'], row['ema200']
    ], dtype=np.float32)


def identify_strategy(row):
    """
    Identify which strategy is applicable based on the row data.
    This is just a placeholder for your strategy identification logic.
    """
    strategy = None
    if row['macd'] > 0 and row['stoch'] < 20:
        strategy = "Bullish Crossover"
    elif row['macd'] < 0 and row['stoch'] > 80:
        strategy = "Bearish Crossover"
    return strategy
