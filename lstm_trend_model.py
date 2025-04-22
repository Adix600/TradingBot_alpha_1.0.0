import torch
import torch.nn as nn

class LSTMTrendModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMTrendModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        out = out[:, -1, :]    # ostatni krok czasowy
        out = self.fc(out)
        return out
