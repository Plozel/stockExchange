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
            nn.Linear(int((1 / 16) * self.hidden_size), 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LSTMModel(nn.Module):

    def __init__(self,num_of_ids, input_size, hidden_size, num_of_layers, is_bidirectional=False):
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_of_layers,
                                 bidirectional=is_bidirectional, batch_first=True)

        # self.id_embedding = nn.Embedding(num_of_ids, config["MLP"]["id_emb_dim"])
        # self.sml_embedding = nn.Embedding(3, config["MLP"]["sml_emb_dim"])

        self.linear = nn.Linear(10, 3)
    def forward(self, x, y, z):
        # stock_id = self.id_embedding(x)
        # sml = self.sml_embedding(y)
        # print(stock_id.shape)
        # print(sml.shape)
        # print(z.shape)
        #
        # x = torch.cat([stock_id, sml, z], 2)
        x, _ = self.predictor(z)
        x = self.linear(x)
        return x


class Inception(nn.Module):
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
    def __init__(self, num_of_ids):
        super(ConvNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=config["MLP"]["first_layer_kernel"], padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )
        self.id_embedding = nn.Embedding(num_of_ids, config["MLP"]["id_emb_dim"])
        self.sml_embedding = nn.Embedding(3, config["MLP"]["sml_emb_dim"])

        self.a3 = Inception(30, 10,  4, 12, 4, 8, 8)
        self.b3 = Inception(38, 14,  6, 16, 4, 10, 10)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(50, 20,  8, 20, 4, 12, 12)
        self.b4 = Inception(64, 22,  9, 24, 4, 14, 16)

        self.a5 = Inception(76, 26,  12, 28, 4, 18, 18)
        self.b5 = Inception(90, 34,  16, 36, 6, 20, 20)

        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(53240, 9)

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
        # print(x.shape)
        x = self.linear(x)

        return self.softmax(x)


class ConvNetFactor(nn.Module):
    def __init__(self, num_of_ids):
        super(ConvNetFactor, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=config["MLP"]["first_layer_kernel"], padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )
        self.id_embedding = nn.Embedding(num_of_ids, config["MLP"]["id_emb_dim"])
        self.sml_embedding = nn.Embedding(3, config["MLP"]["sml_emb_dim"])

        self.a3 = Inception(30, 10,  4, 12, 4, 8, 8)
        self.b3 = Inception(38, 14,  6, 16, 4, 10, 10)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(50, 20,  8, 20, 4, 12, 12)
        self.b4 = Inception(64, 22,  9, 24, 4, 14, 16)

        self.a5 = Inception(76, 26,  12, 28, 4, 18, 18)
        self.b5 = Inception(90, 34,  16, 36, 6, 20, 20)

        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(58190, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, stock_id, sml,  x):
        stock_id = self.id_embedding(stock_id)
        sml = self.sml_embedding(sml)
        x = torch.cat([stock_id, sml, x], 1)
        x = [x.unsqueeze(1) for i in range(x.shape[1])]
        x = torch.cat(x, 1).unsqueeze(1)
        #################################################################
        #                       X1                                      #
        #################################################################
        x1 = self.first_layer(x)
        x1 = self.a3(x1)
        x1 = self.b3(x1)
        x1 = self.maxpool(x1)
        x1 = self.a4(x1)
        x1 = self.b4(x1)
        x1 = self.maxpool(x1)
        x1 = self.a5(x1)
        x1 = self.b5(x1)
        x1 = self.dropout(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        #print("tom",x.shape)
        x2 = self.linear(x1)
        x1 = self.linear(x1)
        """
        #################################################################
        #                       X2                                      #
        #################################################################
        x2 = self.first_layer(x)
        x2 = self.a3(x2)
        x2 = self.b3(x2)
        x2 = self.maxpool(x2)
        x2 = self.a4(x2)
        x2 = self.b4(x2)
        x2 = self.maxpool(x2)
        x2 = self.a5(x2)
        x2 = self.b5(x2)
        x2 = self.dropout(x2)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        #print("tom",x.shape)
        x2 = self.linear(x2)"""
        return self.softmax(x1), self.softmax(x2)