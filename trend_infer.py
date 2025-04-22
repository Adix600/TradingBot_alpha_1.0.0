import torch
import numpy as np
from lstm_data_loader_h1 import prepare_lstm_sequences

def predict_trend_score(df, lstm_model, window=60):
    if len(df) < window + 1:
        return 0.0  # fallback, za mało danych

    df_window = df.tail(window + 1).copy()
    df_window['Sentiment'] = df_window['Sentiment'].rolling(window=3, min_periods=1).mean().fillna(0)

    try:
        X, _, _ = prepare_lstm_sequences(df_window[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']], window=window)
        input_tensor = torch.tensor(X[-1:], dtype=torch.float)
        with torch.no_grad():
            pred = lstm_model(input_tensor).item()
        return float(pred)
    except Exception as e:
        print(f"[TrendPredict] Błąd predykcji trendu: {e}")
        return 0.0
