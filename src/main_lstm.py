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
from torchtext import data
from torchtext.data import Iterator, BucketIterator
# from kornia.losses import FocalLoss

from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset, StockExchangeSeqDataset
from Utils.utils import print_plots, FocalLoss
from models import MLPModel, ConvNet, LSTMModel

torch.set_printoptions(threshold=5000)

if __name__ == '__main__':
    config = yaml.safe_load(open("config.yaml"))

    train = StockExchangeSeqDataset(config["Data"]["train_set"])
    test = StockExchangeSeqDataset(config["Data"]["test_set"], train.id_to_idx)
    num_of_ids = len(train.id_to_idx)

    train_size = len(train)
    test_size = len(test)
    train_loader = DataLoader(train, config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(num_of_ids, config["MLP"]["input_size"], 10, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["MLP"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=True)
    criterion = nn.CrossEntropyLoss().to(device)
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
    os.mkdir(directory_path + "/pickles")
    os.mkdir(directory_path + "/figures")

    for epoch in range(config["MLP"]["num_of_epochs"]):

        model.train()
        predictions = []
        real_values = []
        printable_loss = 0
        correct = 0
        num_of_obs = 0
        i = 0
        for train_batch in tqdm(train_loader):
            stocks_id, sml, features, real_y_1, lengths = train_batch[0].to(device), train_batch[1].to(device), \
                                                           train_batch[2].to(device), train_batch[3].to(device), \
                                                           train_batch[4].to(device)

            stocks_id = stocks_id[:, :max(lengths)]
            sml = sml[:, :max(lengths)]
            features = features[:, :max(lengths), :]
            real_y_1 = real_y_1[:, :max(lengths)]

            model.zero_grad()

            output = model(stocks_id, sml, features)
            output = output.view(-1, 3)
            print(output)
            exit()
            real_y_1 = torch.flatten(real_y_1)

            _, predicted = torch.max(output.data, 1)

            loss = criterion(output, real_y_1)
            printable_loss += loss.item()
            loss.backward()
            # To watch the gradient flow:
            # 1) plot_grad_flow(model.named_parameters())
            optimizer.step()
            correct += (predicted == real_y_1).sum()
            num_of_obs += len(real_y_1)
            predictions.extend(predicted.tolist())

            i += 1
        # 2) plt.show()

        predictions = pd.Series(predictions)
        print("train predictions distribution:")
        print(predictions.value_counts(normalize=True))

        train_acc_list.append(correct.item() / num_of_obs)
        train_loss_list.append(printable_loss / i)

        # save the model
        with open(r"{}/epoch_{}.pkl".format(directory_path + "/pickles", epoch), "wb") as output_file:
            torch.save(model.state_dict(), output_file)

        model.eval()
        printable_loss = 0
        correct = 0
        i = 0
        num_of_obs = 0
        predictions = []

        for test_batch in tqdm(test_loader):
            with torch.no_grad():
                stocks_id, sml, features, real_y_1, lengths = test_batch[0].to(device), test_batch[1].to(device), \
                                                              test_batch[2].to(device), test_batch[3].to(device), \
                                                              test_batch[4].to(device)

                stocks_id = stocks_id[:, :max(lengths)]
                sml = sml[:, :max(lengths)]
                features = features[:, :max(lengths), :]
                real_y_1 = real_y_1[:, :max(lengths)]

                model.zero_grad()

                output = model(stocks_id, sml, features).to(device)
                output = output.view(-1, 3)
                real_y_1 = torch.flatten(real_y_1)

                # threshold = config["MLP"]["majority_threshold"]
                # for i in range(output.shape[0]):
                #     if output[i, 2] < threshold:
                #         output[i, 2] = 0
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
        test_loss_list.append(printable_loss / i)

        if test_acc_list[-1] > max_test_acc:
            max_test_acc = test_acc_list[-1]
            epoch_test_max = epoch

        print("Let's get a taste of the result:")
        print("prediction:", predictions[:10])
        print("real values:", real_values[:10])
        print("Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{} \tTest Loss:{}, \t Test ACC:{}".format(epoch + 1,
                                                                                                         train_loss_list[
                                                                                                             -1],
                                                                                                         train_acc_list[
                                                                                                             -1],
                                                                                                         test_loss_list[
                                                                                                             -1],
                                                                                                         test_acc_list[
                                                                                                             -1]))

        predictions = pd.Series(predictions)
        print("test predictions distribution:")
        print(predictions.value_counts(normalize=True))

        print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, directory_path, time_id)

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






