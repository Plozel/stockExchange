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

    train = StockExchangeDataset("train_class")
    test = StockExchangeDataset("test_class")
    train_size = len(train)
    test_size = len(test)
    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    start_time = timer()

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    max_test_acc = 0  # 10000 is just a random big number
    epoch_test_max = 0

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
        i = 0
        correct = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            model.zero_grad()

            output = model(features).to(device)

            loss = criterion(output, real_y_1-1)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == (real_y_1-1)).sum()
            printable_loss += loss.item()
            i = i + 1
            loss.backward()
            optimizer.step()
        train_acc_list.append(correct.item()/train_size)
        mean_loss = printable_loss/i
        train_loss_list.append(mean_loss)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path+"/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)

        printable_loss = 0
        model.eval()
        i = 0
        correct = 0
        for test_batch in tqdm(test_loader):  # TODO change back to val and create a new evaluate for the test
            with torch.no_grad():
                features, real_y_1, real_y_2 = test_batch[0].to(device), test_batch[1].to(device), test_batch[2].to(device)
                output = model(features)
                loss = criterion(output, real_y_1 - 1)

                _, predicted = torch.max(output.data, 1)
                correct += (predicted == (real_y_1 - 1)).sum()
                printable_loss += loss.item()
                i = i + 1
                predictions.extend((predicted+1).tolist())
                real_values.extend(real_y_1.tolist())

        scheduler.step(printable_loss)

        test_loss_list.append(printable_loss/i)
        test_acc_list.append(correct.item()/test_size)

        if test_acc_list[-1] > max_test_acc:
            max_test_acc = test_acc_list[-1]
            epoch_test_max = epoch

        print("Let's get a taste of the result:")
        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{} \tTest Loss:{}, \t Test ACC:{}".format(epoch + 1, train_loss_list[-1], train_acc_list[-1], test_loss_list[-1], test_acc_list[-1]))

    print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, directory_path, time_id)
    end_time = timer()
    print("the training took: {} sec ".format(round(end_time - start_time, 2)))

    with open('parser_settings.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_id, max_test_acc, epoch_test_max, config["MLP"]["num_of_epochs"], config["MLP"]["batch_size"], config["MLP"]["lr"]])





