import torch
import torch.nn as nn
import torch.optim as optim
from lstm_data_loader_h1 import load_lstm_data_from_mt5
from lstm_trend_model import LSTMTrendModel

def train_lstm_trend_model(symbol='EURUSD', timeframe='H1', months=12, window=60,
                           hidden_size=64, num_layers=2, epochs=10, lr=1e-3):

    X, y, _ = load_lstm_data_from_mt5(symbol, timeframe, months, window)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMTrendModel(input_size=7, hidden_size=hidden_size,
                           num_layers=num_layers, output_size=1).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"[{epoch+1}/{epochs}] Loss: {loss.item():.5f}")

    torch.save(model.state_dict(), "models/lstm_trend_model.pth")
    print("[✓] Model zapisany jako models/lstm_trend_model.pth")

if __name__ == "__main__":
    train_lstm_trend_model()
