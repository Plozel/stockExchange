import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(68, 300),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(300, 68),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(68, 1),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_of_layers, is_bidirectional=False):
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_of_layers,
                                 bidirectional=is_bidirectional, batch_first=True)

    def forward(self, x):
        x = self.predictor(x)

        return x

