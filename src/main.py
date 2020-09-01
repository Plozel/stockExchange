import os
import yaml
import csv
from timeit import default_timer as timer
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from Utils.utils import print_plots
from models import MLPModel


if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))

    train_dataset = StockExchangeDataset("train_data")
    test = StockExchangeDataset("test_data")

    dataset_size = len(train_dataset)
    train_size = int(config["Data"]["train_percent"] * dataset_size)
    train, val = random_split(train_dataset, [train_size, dataset_size - train_size])
    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, config["MLP"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test, config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    criterion = nn.MSELoss()

    start_time = timer()

    train_loss_list = []
    test_loss_list = []
    train_mae_list = []
    test_mae_list = []

    min_test_mae = 10000  # 10000 is just a random big number
    epoch_test_min = 0

    time_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    directory_path = r"../TrainedModels/{}".format(time_id)
    os.mkdir(directory_path)
    os.mkdir(directory_path+"/pickles")
    os.mkdir(directory_path+"/figures")

    for epoch in range(config["MLP"]["num_of_epochs"]):
        model.train()
        predictions = []
        real_values = []
        printable_loss = 0
        train_mae = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            output = model(features)
            train_mae += sum(torch.abs(output[:, 0] - real_y_1).tolist())
            loss = criterion(output[:, 0], real_y_1)
            printable_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        mean_loss = printable_loss/train_size
        train_loss_list.append(mean_loss)
        train_mae_list.append(train_mae/train_size)

        scheduler.step(mean_loss)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path+"/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)

        printable_loss = 0
        test_mae = 0
        model.eval()

        for test_batch in tqdm(test_loader):  # TODO change back to val and create a new evaluate for the test
            with torch.no_grad():
                features, real_y_1, real_y_2 = test_batch[0].to(device), test_batch[1].to(device), test_batch[2].to(device)
                output = model(features)
                test_mae += sum(torch.abs(output[:, 0] - real_y_1).tolist())

                loss = criterion(output[:, 0], real_y_1)

                printable_loss += loss.item()
                predictions.extend(output[:, 0].tolist())

                real_values.extend(real_y_1.tolist())

        test_loss_list.append(printable_loss/len(test))
        test_mae_list.append(test_mae/len(test))
        if test_mae_list[-1] < min_test_mae:
            min_test_mae = test_mae_list[-1]
            epoch_test_min = epoch

        print("Let's get a taste of the result:")
        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train MAE:{} \tTest Loss:{}, \t Test MAE:{}".format(epoch + 1, train_loss_list[-1], train_mae_list[-1], test_loss_list[-1], test_mae_list[-1]))

    print_plots(train_mae_list, train_loss_list, test_mae_list, test_loss_list, directory_path, time_id)
    end_time = timer()
    print("the training took: {} sec ".format(round(end_time - start_time, 2)))

    with open('parser_settings.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_id, min_test_mae, epoch_test_min, config["MLP"]["num_of_epochs"], config["MLP"]["batch_size"], config["MLP"]["lr"]])





