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

from Utils.stock_dataset import StockExchangeDataset
from Utils.utils import print_plots, FocalLoss, box_print
from models import MLPModel, ConvNet


class MainClassifier:

    def __init__(self, target_name, model="conv"):
        # config.yaml control the settings and other hyper parameters
        self.config = yaml.safe_load(open("config.yaml"))

        print("\nBuilding the train dataset:")
        self.train = StockExchangeDataset(self.config["Data"]["train_set"], target_name)
        print("\nBuilding the test dataset:")
        # the test use the id to idx of the train
        self.test = StockExchangeDataset(self.config["Data"]["test_set"], target_name, self.train.id_to_idx)
        print('\n')
        num_of_ids = len(self.train.id_to_idx)

        self.train_loader = DataLoader(self.train, self.config["MLP"]["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test, self.config["MLP"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

        # set the model and it's utils
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model == "conv":
            self.model = ConvNet(num_of_ids).to(self.device)
        else:
            if model == "mlp":
                self.model = MLPModel(num_of_ids).to(self.device)
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
        self.directory_path = r"../TrainedModels/{}".format(self.time_id)
        os.mkdir(self.directory_path)
        os.mkdir(self.directory_path + "/pickles")
        os.mkdir(self.directory_path + "/figures")

    def run_test(self):

        print("Running a test phase:")
        self.model.eval()

        # initialize trackers per epoch
        real_values = []
        predictions = []
        predictions_with_confidence = []
        printable_loss = 0
        num_of_correct = 0
        num_of_correct_with_confidence = 0
        num_of_predicted_with_confidence = 0
        num_of_obs = 0

        i = 0
        for test_batch in tqdm(self.test_loader):
            with torch.no_grad():
                stocks_id, sml, features, true_labels = test_batch[0].to(self.device), \
                                                        test_batch[1].to(self.device), \
                                                        test_batch[2].to(self.device), \
                                                        test_batch[3].to(self.device)

                output = self.model(stocks_id, sml, features).to(self.device)
                loss = self.criterion(output, true_labels)
                printable_loss += loss.item()
                # extracting predictions
                predicted_prob, predicted = torch.max(output.data, 1)

                # gathering the utils for the trackers (acc/confident acc)
                num_of_correct += (predicted == true_labels).sum()
                correct_probs = predicted_prob[predicted == true_labels]
                num_of_correct_with_confidence += len(correct_probs[correct_probs > self.threshold])
                num_of_predicted_with_confidence += len(predicted_prob[predicted_prob > self.threshold])
                num_of_obs += len(true_labels)
                predictions.extend(predicted.tolist())
                real_values.extend(true_labels.tolist())
                correct_pred = predicted[predicted == true_labels]
                predictions_with_confidence.extend(correct_pred[correct_probs > self.threshold].tolist())

                i += 1

        self.scheduler.step(printable_loss)
        self.test_confident_acc_list.append(num_of_correct_with_confidence / (num_of_predicted_with_confidence + 0.0001))

        self.test_acc_list.append(num_of_correct.item() / num_of_obs)
        self.test_loss_list.append(printable_loss / i)

        if self.test_acc_list[-1] > self.max_test_acc:
            self.max_test_acc = self.test_acc_list[-1]
            self.epoch_test_max = self.current_epoch

        print("\nLet's get a taste of the result:")
        print("prediction:", predictions[:100])
        print("real values:", real_values[:100])

        predictions = pd.Series(predictions)
        predictions_with_confidence = pd.Series(predictions_with_confidence)

        print("\n=================================================================")
        print("test predictions distribution:\n{}\n".format(predictions.value_counts(normalize=True)))
        print("test predictions with confidence distribution:\n{}\n".format(predictions_with_confidence.value_counts(normalize=True)))
        print("number of predicted with confidence observations in the test is:{}\n".format(num_of_predicted_with_confidence))
        print("=================================================================\n")

    def run_train(self):
        print("Running the train phase:")

        for epoch in range(self.config["MLP"]["num_of_epochs"]):

            print("\nRuns epoch {}:".format(epoch))
            self.model.train()

            self.current_epoch = epoch

            # initialize trackers per epoch
            predictions = []
            predictions_with_confidence = []
            num_of_correct_with_confidence = 0
            num_of_predicted_with_confidence = 0
            printable_loss = 0
            num_of_correct = 0
            num_of_obs = 0

            i = 0
            for train_batch in tqdm(self.train_loader):
                stocks_id, sml, features, true_labels = train_batch[0].to(self.device),\
                                                               train_batch[1].to(self.device),\
                                                               train_batch[2].to(self.device),\
                                                               train_batch[3].to(self.device)

                self.model.zero_grad()

                output = self.model(stocks_id, sml, features)
                # extracting predictions
                predicted_prob, predicted = torch.max(output.data, 1)

                loss = self.criterion(output, true_labels)
                printable_loss += loss.item()
                loss.backward()
                # To watch the gradient flow:
                # 1) plot_grad_flow(model.named_parameters())
                self.optimizer.step()

                # gathering the utils for the trackers (acc/confident acc)
                num_of_correct += (predicted == true_labels).sum()
                correct_probs = predicted_prob[predicted == true_labels]
                num_of_correct_with_confidence += len(correct_probs[correct_probs > self.threshold])
                num_of_predicted_with_confidence += len(predicted_prob[predicted_prob > self.threshold])
                num_of_obs += len(true_labels)
                predictions.extend(predicted.tolist())
                correct_pred = predicted[predicted == true_labels]
                predictions_with_confidence.extend(correct_pred[correct_probs > self.threshold].tolist())

                i += 1

            # 2) plt.show()

            predictions = pd.Series(predictions)
            predictions_with_confidence = pd.Series(predictions_with_confidence)
            print("\n=================================================================")
            print("Current train predictions distribution:\n{}\n".format(predictions.value_counts(normalize=True)))
            print("Current train confident predictions distribution:\n{}\n".format(predictions_with_confidence.value_counts(normalize=True)))
            print("number of predicted with confident observations in the train is:{}".format(num_of_predicted_with_confidence))
            print("=================================================================\n")

            self.train_confident_acc_list.append(num_of_correct_with_confidence / (num_of_predicted_with_confidence + 0.001))
            self.train_acc_list.append(num_of_correct.item() / num_of_obs)
            self.train_loss_list.append(printable_loss / i)

            # saves the model
            with open(r"{}/epoch_{}.pkl".format(self.directory_path + "/pickles", epoch), "wb") as output_file:
                torch.save(self.model.state_dict(), output_file)

            # test the current model
            self.run_test()

            print(
                "Epoch:{} Completed,\tTrain Loss:{},\t Train ACC:{},\t Train confident ACC:{} \tTest Loss:{},"
                " \t Test ACC:{},\t test confident ACC:{}".format(self.current_epoch + 1, self.train_loss_list[-1],
                                                                  self.train_acc_list[-1],
                                                                  self.train_confident_acc_list[-1],
                                                                  self.test_loss_list[-1], self.test_acc_list[-1],
                                                                  self.test_confident_acc_list[-1]))

            print_plots(self.train_acc_list, self.train_loss_list, self.test_acc_list, self.test_loss_list, self.directory_path, self.time_id)

            with open('{}/log{}.csv'.format(self.directory_path, self.time_id), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.time_id, self.max_test_acc, self.epoch_test_max, self.config["MLP"]])

        end_time = timer()
        print("Finish training, it took: {} sec ".format(round(end_time - self.start_time, 2)))

        return self.time_id, self.max_test_acc, self.epoch_test_max, self.directory_path, self.start_time

    def predict(self, stocks_id, sml, features):

        output = self.model(stocks_id, sml, features).to(self.device)
        predicted_prob, predicted = torch.max(output.data, 1)

        return predicted_prob, predicted

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
