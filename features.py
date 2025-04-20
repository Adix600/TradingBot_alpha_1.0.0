import MetaTrader5 as mt5
import pandas as pd


def get_ema200(symbol):
    """
    Funkcja pobiera dane z MetaTrader5 i oblicza EMA200 dla danego symbolu.
    Zwraca wartość EMA200 na ostatniej świecy.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)  # Pobierz ostatnie 200 świec
    if not rates:
        raise ValueError(f"Nie udało się pobrać danych dla {symbol} z MetaTrader 5.")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ema200'] = df['close'].ewm(span=200).mean()  # Oblicz EMA200
    return df['ema200'].iloc[-1]  # Zwróć EMA200 na ostatniej świecy


def prepare_data(df, sentiment):
    """
    Funkcja przygotowuje dane dla modelu, w tym wskaźniki techniczne i sentiment.
    """
    df = df.copy()

    # Wskaźniki techniczne
    df['ma_short'] = df['close'].rolling(5).mean()
    df['ma_long'] = df['close'].rolling(15).mean()
    df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
    df['volatility'] = df['close'].rolling(10).std()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['stoch'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / (
                df['high'].rolling(14).max() - df['low'].rolling(14).min()))

    # Dodajemy EMA200
    symbol = df['symbol'][0]  # Zakładając, że każda linia w DataFrame ma zdefiniowany symbol
    df['ema200'] = get_ema200(symbol)

    df = df.dropna()

    # Normalizacja danych
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = ['ma_short', 'ma_long', 'rsi', 'volatility', 'momentum', 'macd', 'stoch', 'ema200']
    df[features] = scaler.fit_transform(df[features])

    # Dodanie sentymentu
    df['sentiment'] = sentiment

    return df


def get_state(row):
    """
    Funkcja przekształca jeden wiersz danych na wektor cech.
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
    # Implement your strategy logic here
    # Example: Check if the MACD and Stochastic indicators align for a trade
    strategy = None
    if row['macd'] > 0 and row['stoch'] < 20:
        strategy = "Bullish Crossover"
    elif row['macd'] < 0 and row['stoch'] > 80:
        strategy = "Bearish Crossover"
    return strategy
