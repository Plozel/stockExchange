
import yaml

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from models import MLPModel

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.MSELoss()

    for epoch in range(config["MLP"]["num_of_epochs"]):
        model.train()
        train_loss_list = []
        val_loss_list = []
        train_mae_list = []
        val_mae_list = []
        printable_loss = 0
        train_mae = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            output = model(features)
            train_mae += torch.abs(output - real_y_1).item()

            loss = criterion(output, real_y_1)
            printable_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        mean_loss = printable_loss/train_size
        train_loss_list.append(mean_loss)
        train_mae_list.append(train_mae/train_size)

        scheduler.step(mean_loss)

        printable_loss = 0
        val_mae = 0
        model.eval()
        predictions = []
        real_values = []
        for val_batch in tqdm(val_loader):
            with torch.no_grad():
                features, real_y_1, real_y_2 = val_batch[0].to(device), val_batch[1].to(device), val_batch[2].to(device)
                output = model(features)
                val_mae += torch.abs(output - real_y_1).item()
                loss = criterion(output, real_y_1)
                printable_loss += loss.item()
                predictions.append(output.item())
                real_values.append(real_y_1.item())

        val_loss_list.append(printable_loss/len(val))
        val_mae_list.append(val_mae/len(val))

        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train MAE:{} \tValidation Loss:{}, \t Validation MAE:{}".format(epoch + 1, train_loss_list[-1], train_mae_list[-1], val_loss_list[-1], val_mae_list[-1]))






