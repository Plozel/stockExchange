
import yaml

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from Models.MLP import MLPModel

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))

    dataset = StockExchangeDataset()
    dataset_size = len(dataset)
    train_size = int(config["Data"]["train_percent"] * dataset_size)
    train, val = random_split(dataset, [train_size, dataset_size - train_size])

    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["MLP"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["MLP"]["lr"])
    criterion = nn.MSELoss()

    for epoch in range(config["MLP"]["num_of_epochs"]):
        model.train()
        loss_list = []
        printable_loss = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            output = model(features)
            loss = criterion(output, real_y_1)
            printable_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()




