import torch
import torch.nn as nn
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Ekstraktor cech z LSTM
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, hidden_size=64, dropout=0.2):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[1]
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, features_dim)

    def forward(self, observations):
        x = observations  # już [batch, seq, features]
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        return self.linear(x)


# Polityka z LSTM, zgodna z RecurrentPPO
class CustomLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        self.features_extractor_class = LSTMFeatureExtractor
        self.features_extractor_kwargs = kwargs.get("features_extractor_kwargs", {
            "features_dim": 128, "hidden_size": 64, "dropout": 0.2
        })

        # Ustaw domyślne wartości jeśli nie zostały przekazane
        kwargs.setdefault("features_extractor_class", self.features_extractor_class)
        kwargs.setdefault("features_extractor_kwargs", self.features_extractor_kwargs)

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

# Loader modelu trendowego LSTM
def load_lstm_trend_model(path="models/lstm_trend_model.pth", input_size=6):
    from lstm_trend_model import LSTMTrendModel
    model = LSTMTrendModel(input_size=input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
