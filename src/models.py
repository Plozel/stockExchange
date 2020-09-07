import torch
import torch.nn as nn
import yaml

config = yaml.safe_load(open("config.yaml"))


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()

        self.hidden_size = config["MLP"]["hidden_size"]
        self.input_size = config["MLP"]["input_size"]

        self.seq = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, int((3/4) * self.hidden_size)),
            nn.Tanh(),
            nn.Linear(int((3 / 4) * self.hidden_size), int((2 / 4) * self.hidden_size)),
            nn.Tanh(),
            nn.Linear(int((2 / 4) * self.hidden_size), int((1 / 4) * self.hidden_size)),
            nn.Tanh(),
            nn.Linear(int((1 / 4) * self.hidden_size), int((1 / 8) * self.hidden_size)),
            nn.Tanh(),
            nn.Linear(int((1 / 8) * self.hidden_size), int((1 / 16) * self.hidden_size)),
            nn.Tanh(),
            nn.Linear(int((1 / 16) * self.hidden_size), 1),
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

