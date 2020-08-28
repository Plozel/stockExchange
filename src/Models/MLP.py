import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(78, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.seq(x)
        return x