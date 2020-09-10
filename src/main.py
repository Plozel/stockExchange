import os
import yaml
import csv
from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from Utils.utils import print_plots
from models import MLPModel, ConvNet

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))

    train = StockExchangeDataset("train_class")
    test = StockExchangeDataset("test_class")
    train_size = len(train)
    test_size = len(test)
    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    # weight = torch.tensor([0.9, 0.65, 1.05])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.99, 0.97, 1.0])).to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    start_time = timer()

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    max_test_acc = 0  # 10000 is just a random big number
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
        correct = 0
        num_of_obs = 0
        i = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            model.zero_grad()
            # real_y_1 = real_y_1.unsqueeze(1)
            output = model(features)
            _, predicted = torch.max(output.data, 1)

            loss = criterion(output, real_y_1)
            printable_loss += loss.item()
            loss.backward()
            # To watch the gradient flow:
            # plot_grad_flow(model.named_parameters())
            optimizer.step()
            correct += (predicted == real_y_1).sum()
            num_of_obs += len(real_y_1)

            i += 1
        # plt.show()

        train_acc_list.append(correct.item() / num_of_obs)
        train_loss_list.append(printable_loss/i)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path+"/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)

        printable_loss = 0
        model.eval()

        correct = 0
        i = 0
        num_of_obs = 0
        for test_batch in tqdm(test_loader):
            with torch.no_grad():
                features, real_y_1, real_y_2 = test_batch[0].to(device), test_batch[1].to(device), test_batch[2].to(device)
                # real_y_1 = real_y_1.unsqueeze(1)
                output = model(features).to(device)
                _, predicted = torch.max(output.data, 1)

                loss = criterion(output, real_y_1)
                printable_loss += loss.item()
                correct += (predicted == real_y_1).sum()
                num_of_obs += len(real_y_1)
                predictions.extend(predicted.tolist())
                real_values.extend(real_y_1.tolist())
                i += 1

        scheduler.step(printable_loss)
        test_acc_list.append(correct.item() / num_of_obs)
        test_loss_list.append(printable_loss/i)

        if test_acc_list[-1] > max_test_acc:
            max_test_acc = test_acc_list[-1]
            epoch_test_max = epoch


        print("Let's get a taste of the result:")
        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{} \tTest Loss:{}, \t Test ACC:{}".format(epoch + 1, train_loss_list[-1], train_acc_list[-1], test_loss_list[-1], test_acc_list[-1]))

        predictions = pd.Series(predictions)
        print(predictions.value_counts(normalize=True))

    print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, directory_path, time_id)
    end_time = timer()
    print("the training took: {} sec ".format(round(end_time - start_time, 2)))

    with open('parser_settings.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_id, max_test_acc, epoch_test_min, config["MLP"]["num_of_epochs"], config["MLP"]["batch_size"], config["MLP"]["lr"]])





