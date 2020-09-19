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
from Utils.utils import print_plots, FocalLoss
from models import MLPModel, ConvNet

from src.models import ConvNetFactor

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))
    train = StockExchangeDataset(config["Data"]["train_set"])
    test = StockExchangeDataset(config["Data"]["test_set"], train.id_to_idx)
    num_of_ids = len({**train.id_to_idx, **test.id_to_idx})

    train_size = len(train)
    test_size = len(test)
    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetFactor(num_of_ids).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=True)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss().to(device)
    start_time = timer()
    threshold = config["MLP"]["majority_threshold"]

    train_loss_list = []
    test_loss_list = []
    train_1_acc_list = []
    print(train_1_acc_list)
    test_1_acc_list = []
    train_2_acc_list = []
    test_2_acc_list = []

    max_test_acc = 0  # 10000 is just a random big number
    epoch_test_max = 0

    time_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    directory_path = r"../TrainedModels/{}".format(time_id)
    os.mkdir(directory_path)
    os.mkdir(directory_path+"/pickles")
    os.mkdir(directory_path+"/figures")

    for epoch in range(config["MLP"]["num_of_epochs"]):

        model.train()
        predictions1 = []
        predictions2 = []
        printable_loss = 0
        correct1 = 0
        correct2 = 0
        num_of_obs = 0
        i = 0
        for train_batch in tqdm(train_loader):
            stocks_id, sml, features, real_y_1, real_y_2, real_y_1_2 = train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device), train_batch[3].to(device), train_batch[4].to(device), train_batch[5].to(device)
            model.zero_grad()
            output1, output2 = model(stocks_id, sml, features)
            predicted_prob1, predicted1 = torch.max(output1.data, 1)
            predicted_prob2, predicted2 = torch.max(output2.data, 1)
            loss1 = criterion(output1, real_y_1)
            loss2 = criterion(output2, real_y_2)
            loss = loss1 + loss2
            printable_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct1 += (predicted1 == real_y_1).sum()
            correct2 += (predicted1 == real_y_2).sum()
            num_of_obs += len(real_y_1)
            predictions1.extend(predicted1.tolist())
            predictions2.extend(predicted2.tolist())
            i += 1

        predictions1 = pd.Series(predictions1)
        predictions2 = pd.Series(predictions2)
        print("#############################################")
        print("train predictions 1 distribution:")
        print(predictions1.value_counts(normalize=True))
        print("train predictions 2 distribution:")
        print(predictions2.value_counts(normalize=True))
        print("#############################################")
        train_1_acc_list.append(correct1.item() / num_of_obs)
        train_2_acc_list.append(correct2.item() / num_of_obs)
        train_loss_list.append(printable_loss/i)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path+"/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)


        model.eval()
        predictions1 = []
        predictions2 = []
        printable_loss = 0
        correct1 = 0
        correct2 = 0
        num_of_obs = 0
        i = 0
        for test_batch in tqdm(test_loader):
            with torch.no_grad():
                stocks_id, sml, features, real_y_1, real_y_2, real_y_1_2 = test_batch[0].to(device), test_batch[1].to(device), \
                                                               test_batch[2].to(device), test_batch[3].to(device), \
                                                               test_batch[4].to(device), test_batch[5].to(device)
                output1, output2 = model(stocks_id, sml, features)
                predicted_prob1, predicted1 = torch.max(output1.data, 1)
                predicted_prob2, predicted2 = torch.max(output2.data, 1)
                loss1 = criterion(output1, real_y_1)
                loss2 = criterion(output2, real_y_2)
                loss = loss1 + loss2
                printable_loss += loss.item()

                correct1 += (predicted1 == real_y_1).sum()
                correct2 += (predicted1 == real_y_2).sum()
                num_of_obs += len(real_y_1)
                predictions1.extend(predicted1.tolist())
                predictions2.extend(predicted2.tolist())
                i += 1


        scheduler.step(printable_loss)
        predictions1 = pd.Series(predictions1)
        predictions2 = pd.Series(predictions2)
        print("#############################################")
        print("test predictions 1 distribution:")
        print(predictions1.value_counts(normalize=True))
        print("test predictions 2 distribution:")
        print(predictions2.value_counts(normalize=True))
        print("#############################################")
        test_1_acc_list.append(correct1.item() / num_of_obs)
        test_2_acc_list.append(correct2.item() / num_of_obs)
        test_loss_list.append(printable_loss/i)



        if test1_acc_list[-1] > max_test_acc:
            max_test_acc = test1_acc_list[-1]
            epoch_test_max = epoch

        print("Epoch:{} Completed,\tTrain Loss:{},\t Train1 ACC:{},\t Train2 ACC:{} \tTest Loss:{}, \t Test1 ACC:{},\t test1 confident ACC:{}".format(epoch + 1, train_loss_list[-1], train_1_acc_list[-1], train_2_acc_list[-1], test_loss_list[-1], test_1_acc_list[-1], test_2_acc_list[-1]))

        predictions1 = pd.Series(predictions1)
        predictions2 = pd.Series(predictions2)
        print("#############################################")
        print("test predictions 1 distribution:")
        print(predictions1.value_counts(normalize=True))
        print("test predictions 2 distribution:")
        print(predictions2.value_counts(normalize=True))
        print("#############################################")

        print_plots(train1_acc_list, train_loss_list, test1_acc_list, test_loss_list, directory_path, time_id)

        with open('{}/log{}.csv'.format(directory_path, time_id), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [time_id, max_test_acc, epoch_test_max, config["MLP"]])

    with open('../TrainedModels/general_log.csv'.format(directory_path, time_id), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, max_test_acc, epoch_test_max, config["MLP"]])

    end_time = timer()
    print("Finish training, it took: {} sec ".format(round(end_time - start_time, 2)))







