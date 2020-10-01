import torch
import torch.nn as nn
import yaml

config = yaml.safe_load(open("config.yaml"))


class MLPModel(nn.Module):
    """Creates a MLP model."""

    def __init__(self, num_of_ids, num_of_classes):
        super(MLPModel, self).__init__()

        self.hidden_size = config["CommonParameters"]["hidden_size"]
        self.input_size = config["CommonParameters"]["input_size"]

        self.id_embedding = nn.Embedding(num_of_ids, config["CommonParameters"]["id_emb_dim"])
        self.sml_embedding = nn.Embedding(3, config["CommonParameters"]["sml_emb_dim"])

        self.seq = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_of_classes),
        )

    def forward(self, stock_id, sml,  x):
        stock_id = self.id_embedding(stock_id)
        sml = self.sml_embedding(sml)
        x = torch.cat([stock_id, sml, x], 1)
        x = self.seq(x)
        return x


class Inception(nn.Module):
    """Creates an Inception module"""

    def __init__(self, in_channels, nof1x1, nof3x3_1, nof3x3_out, nof5x5_1, nof5x5_out, pool_planes):
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.b1x1 = nn.Sequential(
            nn.Conv2d(in_channels, nof1x1, kernel_size=1),
            nn.BatchNorm2d(nof1x1),
            nn.ReLU(),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b1x3 = nn.Sequential(
            nn.Conv2d(in_channels, nof3x3_1, kernel_size=1),
            nn.BatchNorm2d(nof3x3_1),
            nn.ReLU(),
            nn.Conv2d(nof3x3_1, nof3x3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(nof3x3_out),
            nn.ReLU(),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b1x5 = nn.Sequential(
            nn.Conv2d(in_channels, nof5x5_1, kernel_size=1),
            nn.BatchNorm2d(nof5x5_1),
            nn.ReLU(),
            nn.Conv2d(nof5x5_1, nof5x5_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(nof5x5_out),
            nn.ReLU(),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b3x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )

    def forward(self, a):
        b1 = self.b1x1.forward(a)
        b2 = self.b1x3.forward(a)
        b3 = self.b1x5.forward(a)
        b4 = self.b3x1.forward(a)
        return torch.cat([b1, b2, b3, b4], 1)


class ConvNet(nn.Module):
    """Creates a CNN+Inception model."""

    def __init__(self, num_of_ids, num_of_classes):
        super(ConvNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=config["CommonParameters"]["first_layer_kernel"], padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )

        self.num_of_classes = num_of_classes
        self.id_embedding = nn.Embedding(num_of_ids, config["CommonParameters"]["id_emb_dim"])
        self.sml_embedding = nn.Embedding(3, config["CommonParameters"]["sml_emb_dim"])

        self.a3 = Inception(30, 10,  4, 12, 4, 8, 8)
        self.b3 = Inception(38, 14,  6, 16, 4, 10, 10)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(50, 20,  8, 20, 4, 12, 12)
        self.b4 = Inception(64, 22,  9, 24, 4, 14, 16)

        self.a5 = Inception(76, 26,  12, 28, 4, 18, 18)
        self.b5 = Inception(90, 34,  16, 36, 6, 20, 20)

        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.dropout = nn.Dropout(p=0.3)
        self.pre_first_layer = nn.Linear(88, 120)
        self.linear = nn.Linear(68750, self.num_of_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, stock_id, sml,  x):

        stock_id = self.id_embedding(stock_id)
        sml = self.sml_embedding(sml)
        x = torch.cat([stock_id, sml, x], 1)
        x = [x.unsqueeze(1) for i in range(x.shape[1])]
        x = torch.cat(x, 1).unsqueeze(1)
        x = self.first_layer(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return self.softmax(x)

