import os
import yaml
import csv

from timeit import default_timer as timer
from datetime import datetime

import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

from Utils.utils import print_plots, FocalLoss, box_print
from models import MLPModel, ConvNet, PlozNet


class MainClassifier:

    def __init__(self, target_name, model, train_dataset, test_dataset, num_of_classes):
        super().__init__()

        # config.yaml control the settings and other hyper parameters
        self.config = yaml.safe_load(open("config.yaml"))

        self.train = train_dataset
        self.test = test_dataset

        self.num_of_ids = len(self.train.id_to_idx)
        self.num_of_classes = num_of_classes

        self.train_loader = DataLoader(self.train, self.config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test, self.config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)


        # set the model and it's utils
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_name = target_name

        if model == "conv":
            self.model = ConvNet(self.num_of_ids, self.num_of_classes).to(self.device)
        elif model == "PlozNet":
            self.model = PlozNet(self.num_of_ids, self.num_of_classes).to(self.device)
        else:
            if model == "mlp":
                self.model = MLPModel(self.num_of_ids, self.num_of_classes).to(self.device)
            else:
                print("Invalid model")
                exit()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["MLP"]["lr"])

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=30, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # a threshold for the confident acc
        self.threshold = self.config["MLP"]["majority_threshold"]

        # loss/acc trackers through time (epochs)
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_confident_acc_list = []
        self.test_confident_acc_list = []

        self.max_test_acc = 0
        self.epoch_test_max = 0
        self.current_epoch = 0

        self.start_time = timer()
        self.time_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        # creates directories to save trackers utils
        self.directory_path = r"../TrainedModels/{}_{}".format(self.target_name, self.time_id)
        os.mkdir(self.directory_path)
        os.mkdir(self.directory_path + "/pickles")
        os.mkdir(self.directory_path + "/figures")

    def run_test(self):

        print("Running a test phase:")
        self.model.eval()

        # initialize trackers per epoch
        real_values = []
        predictions = []
        printable_loss = 0
        num_of_correct_1 = 0
        num_of_correct_2 = 0
        num_of_obs = 0

        i = 0
        for test_batch in tqdm(self.test_loader):
            with torch.no_grad():
                stocks_id, sml, features, true_labels_1, true_labels_2, true_labels_3 = test_batch[0].to(self.device), \
                                                        test_batch[1].to(self.device), \
                                                        test_batch[2].to(self.device), \
                                                        test_batch[3].to(self.device), \
                                                        test_batch[4].to(self.device), \
                                                        test_batch[5].to(self.device)

                output_1, output_2, output_3 = self.model(stocks_id, sml, features)
                output = torch.zeros(output_1.shpae()[0],
                                     self.config["MLP"]["num_of_classes"] * self.config["MLP"]["num_of_classes"])
                output = output.to(self.device)
                for i in range(len(output)):
                    for j in range((self.config["MLP"]["num_of_classes"]) ** 2):
                        if j == 0:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0]) * (output_2[i][0])
                        elif j == 1:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0]) * (output_2[i][1])
                        elif j == 2:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0]) * (output_2[i][2])
                        elif j == 3:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1]) * (output_2[i][0])
                        elif j == 4:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1]) * (output_2[i][1])
                        elif j == 5:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1]) * (output_2[i][2])
                        elif j == 6:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2]) * (output_2[i][0])
                        elif j == 7:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2]) * (output_2[i][1])
                        elif j == 8:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2]) * (output_2[i][2])
                self.softmax = nn.Softmax(dim=1)
                output = self.softmax(output)
                loss = self.criterion(output, true_labels_3)
                printable_loss += loss.item()
                # extracting predictions
                predicted_prob, predicted = torch.max(output.data, 1)

                for i in range(len(true_labels_1)):
                    num_of_obs += 1
                    if (true_labels_1[i] == 0) and (predicted[i] in(0, 1, 2)):
                        num_of_correct_1 += 1
                    elif (true_labels_1[i] == 1) and (predicted[i] in(3, 4, 5)):
                        num_of_correct_1 += 1
                    elif (true_labels_1[i] == 2) and (predicted[i] in(6, 7, 8)):
                        num_of_correct_1 += 1

                i += 1

        self.scheduler.step(printable_loss)

        self.test_acc_list.append(num_of_correct_1.item() / num_of_obs)
        self.test_loss_list.append(printable_loss / i)



        print("\nLet's get a taste of the result:")

        predictions.extend(predicted.tolist())
        predictions = pd.Series(predictions)
        print("\n=================================================================")
        print("test predictions distribution:\n{}\n".format(predictions.value_counts(normalize=True)))
        print("=================================================================\n")

    def run_train(self):
        print("Running the train phase:")

        for epoch in range(self.config["MLP"]["num_of_epochs"]):

            print("\nRuns epoch {}:".format(epoch))
            self.model.train()

            self.current_epoch = epoch

            # initialize trackers per epoch
            predictions = []
            printable_loss = 0
            num_of_correct_1 = 0
            num_of_correct_2 = 0
            num_of_obs = 0

            i = 0
            for train_batch in tqdm(self.train_loader):
                stocks_id, sml, features, true_labels_1, true_labels_2, true_labels_3 = train_batch[0].to(self.device),\
                                                               train_batch[1].to(self.device),\
                                                               train_batch[2].to(self.device),\
                                                               train_batch[3].to(self.device), \
                                                               train_batch[4].to(self.device),\
                                                               train_batch[5].to(self.device)


                self.model.zero_grad()

                output_1, output_2, output_3 = self.model(stocks_id, sml, features)
                output = torch.zeros(output_1.shpae()[0], self.config["MLP"]["num_of_classes"]*self.config["MLP"]["num_of_classes"])
                output = output.to(self.device)
                for i in range(len(output)):
                    for j in range((self.config["MLP"]["num_of_classes"])**2):
                        if j == 0:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0]) * (output_2[i][0])
                        elif j == 1:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0]) * (output_2[i][1])
                        elif j == 2:
                            output[i][j] = (output_3[i][j]) * (output_1[i][0])*(output_2[i][2])
                        elif j == 3:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1]) * (output_2[i][0])
                        elif j == 4:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1]) * (output_2[i][1])
                        elif j == 5:
                            output[i][j] = (output_3[i][j]) * (output_1[i][1])*(output_2[i][2])
                        elif j == 6:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2]) * (output_2[i][0])
                        elif j == 7:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2]) * (output_2[i][1])
                        elif j == 8:
                            output[i][j] = (output_3[i][j]) * (output_1[i][2])*(output_2[i][2])
                self.softmax = nn.Softmax(dim=1)
                output = self.softmax(output)

                # extracting predictions
                predicted_prob, predicted = torch.max(output.data, 1)

                loss = self.criterion(output, true_labels_3)
                printable_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                # Gathers utils for the trackers (acc/confident acc)
                for i in range(len(true_labels_1)):
                    num_of_obs += 1
                    if (true_labels_1[i] == 0) and (predicted[i] in(0, 1, 2)):
                        num_of_correct_1 += 1
                    elif (true_labels_1[i] == 1) and (predicted[i] in(3, 4, 5)):
                        num_of_correct_1 += 1
                    elif (true_labels_1[i] == 2) and (predicted[i] in(6, 7, 8)):
                        num_of_correct_1 += 1


                i += 1

            predictions.extend(predicted.tolist())
            predictions = pd.Series(predictions)
            print("\n=================================================================")
            print("Current train predictions distribution:\n{}\n".format(predictions.value_counts(normalize=True)))
            print("=================================================================\n")

            self.train_acc_list.append(num_of_correct_1.item() / num_of_obs)
            self.train_loss_list.append(printable_loss / i)


            # test the current model
            #self.run_test()
            print(
                "Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{}".format(self.current_epoch + 1, self.train_loss_list[-1],
                                                                  self.train_acc_list[-1]))
            """print(
                "Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{},\t Train confident ACC:{} \tTest Loss:{},"
                " \t Test ACC:{},\t Test confident ACC:{}".format(self.current_epoch + 1, self.train_loss_list[-1],
                                                                  self.train_acc_list[-1],
                                                                  self.train_confident_acc_list[-1],
                                                                  self.test_loss_list[-1], self.test_acc_list[-1],
                                                                  self.test_confident_acc_list[-1]))"""




        end_time = timer()
        print("Finish training, it took: {} sec ".format(round(end_time - self.start_time, 2)))


    def predict(self, stocks_id, sml, features):

        output = self.model(stocks_id, sml, features).to(self.device)
        predicted_prob, predicted = torch.max(output.data, 1)

        return predicted_prob, predicted

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
