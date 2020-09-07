import os
import yaml
import csv
from timeit import default_timer as timer
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from Utils.utils import print_plots
from models import MLPModel

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True)
    criterion = nn.MSELoss().to(device)
    criterion2 = nn.L1Loss().to(device)
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
        current_mae = 0
        i = 0
        for train_batch in tqdm(train_loader):
            features, real_y_1, real_y_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device)
            model.zero_grad()
            real_y_1 = real_y_1.unsqueeze(1)

            output = model(features)
            loss = criterion(output, real_y_1)
            printable_loss += loss.item()
            loss.backward()
            # plot_grad_flow(model.named_parameters())
            #g
            optimizer.step()
            current_mae += criterion2(output, real_y_1)
            i += 1
        # plt.show()

        train_mae_list.append(current_mae.item()/i)
        train_loss_list.append(printable_loss/i)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path+"/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)

        printable_loss = 0
        model.eval()

        current_mae = 0
        for test_batch in tqdm(test_loader):  # TODO change back to val and create a new evaluate for the test
            with torch.no_grad():
                features, real_y_1, real_y_2 = test_batch[0].to(device), test_batch[1].to(device), test_batch[2].to(device)
                real_y_1 = real_y_1.unsqueeze(1)
                output = model(features).to(device)
                loss = criterion(output, real_y_1)
                printable_loss += loss.item()
                current_mae += criterion2(output, real_y_1)
                predictions.extend(output.tolist())
                real_values.extend(real_y_1.tolist())

        scheduler.step(printable_loss)
        test_mae_list.append(current_mae.item()/test_loader.__len__())
        test_loss_list.append(printable_loss/test_loader.__len__())

        if test_mae_list[-1] < min_test_mae:
            min_test_mae = test_mae_list[-1]
            epoch_test_min = epoch

        print("Let's get a taste of the result:")
        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{} \tTest Loss:{}, \t Test ACC:{}".format(epoch + 1, train_loss_list[-1], train_mae_list[-1], test_loss_list[-1], test_mae_list[-1]))

    print_plots(train_mae_list, train_loss_list, test_mae_list, test_loss_list, directory_path, time_id)
    end_time = timer()
    print("the training took: {} sec ".format(round(end_time - start_time, 2)))

    with open('parser_settings.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([time_id, min_test_mae, epoch_test_min, config["MLP"]["num_of_epochs"], config["MLP"]["batch_size"], config["MLP"]["lr"]])





