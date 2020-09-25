import yaml
import csv
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from Utils.stock_dataset import StockExchangeDataset
from classifiers import MainClassifier


def run_training(classifier):

    print("Trains to classify {}".format(classifier.target_name))
    print("--------------------")
    time_id, max_test_acc, epoch_test_max = classifier.run_train()

    with open('../TrainedModels/general_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, classifier.target_name, max_test_acc, epoch_test_max, *config["MLP"]])


def train_all(_class_1_classifier, _class_2_classifier, _class_3_classifier):

    run_training(_class_1_classifier)
    run_training(_class_2_classifier)
    run_training(_class_3_classifier)


def load_all(_class_1_classifier, _class_2_classifier, _class_3_classifier):

    print("-----------------------")
    print("Loading saved models:")
    print("-----------------------")

    _class_1_classifier.load_model(config["TrainedModels"]["pkl_path_1"])
    _class_2_classifier.load_model(config["TrainedModels"]["pkl_path_2"])
    _class_3_classifier.load_model(config["TrainedModels"]["pkl_path_3"])


def test_all_separate(_class_1_classifier, _class_2_classifier, _class_3_classifier):
    print("-----------------------")
    print("Test all models:")
    print("-----------------------")

    _class_1_classifier.run_test()
    _class_2_classifier.run_test()
    _class_3_classifier.run_test()

def xgboost_classifiers():
    df_train = pd.read_csv("../data/train_class_1_2_3.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    df_test = pd.read_csv("../data/test_class_1_2_3.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    X = df_train.iloc[:, np.r_[36:79, 11:26]].to_numpy()
    y_1 = df_train.loc[:, 'class_1'].to_numpy()
    y_2 = df_train.loc[:, 'class_2'].to_numpy()
    X_test = df_test.iloc[:, np.r_[36:79, 11:26]].to_numpy()
    y_test = df_test.loc[:, 'class_1'].to_numpy()
    print("xgbingggg")
    xgb_model_1 = xgb.XGBRFClassifier(random_state=0, n_estimators=5, max_depth=10)
    xgb_model_1.fit(X, y_1)
    print(xgb_model_1.score(X_test, y_test))
    #xgb_model_2 = xgb.XGBRFClassifier(random_state=42, n_estimators=100, max_depth=10)
    #xgb_model_2.fit(X, y_2)
    return xgb_model_1#, xgb_model_2
def test_all_together(_class_1_classifier, _class_2_classifier, _class_3_classifier):
    #xgb_model_1, xgb_model_2 = xgboost_classifiers()
    xgb_model_1 = xgboost_classifiers()
    print("-----------------------")
    print("Test all models:")
    print("-----------------------")
    num_of_classes = 3
    test_loader = _class_1_classifier.test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _class_1_classifier.model.eval()
    _class_2_classifier.model.eval()
    _class_3_classifier.model.eval()

    num_of_correct_1 = 0
    num_of_correct_2 = 0
    num_of_obs = 0
    num_tom_correct = 0
    num_tom_obs = 0
    for test_batch in tqdm(test_loader):
        with torch.no_grad():
            stocks_id, sml, features, true_labels_1, true_labels_2, true_labels_3 = test_batch[0].to(device), \
                                                                                    test_batch[1].to(device), \
                                                                                    test_batch[2].to(device), \
                                                                                    test_batch[3].to(device), \
                                                                                    test_batch[4].to(device), \
                                                                                    test_batch[5].to(device)

            xgb_boost_1_prob = torch.from_numpy(xgb_model_1.predict_proba(features.cpu().numpy())).to(device)
            xgb_boost_1_labels = torch.from_numpy(xgb_model_1.predict(features.cpu().numpy())).to(device)
            #print(xgb_model_1.score(features.cpu().detach().numpy(), true_labels_1.cpu().detach().numpy()))
            num_tom_correct += xgb_model_1.score(features.cpu().numpy(), true_labels_1.cpu().numpy())*len(true_labels_1.cpu().numpy())
            num_tom_obs += len(true_labels_1.cpu().numpy())
            print("tommm ",num_tom_obs)
            print(num_tom_correct/num_tom_obs)
            #print(torch.max(xgb_boost_1_prob.data, 1)[1])
            #print(features.cpu().numpy().shape)
            outputs = [_class_1_classifier.model(stocks_id, sml, features).to(device),
                       _class_2_classifier.model(stocks_id, sml, features).to(device),
                       _class_3_classifier.model(stocks_id, sml, features).to(device)]

            net_1_prob = torch.max(outputs[0].data, 1)[0]
            net_1_labels = torch.max(outputs[0].data, 1)[1]
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #print(torch.max(xgb_boost_1_prob.data, 1)[0])
            #print("#############################################")
            #print(net_1_prob)
            #print(net_1_labels)

            """predict_comb = (0*outputs[0])+(1*(xgb_boost_1_prob))
            num_tom_correct += int(((torch.max(predict_comb.data, 1)[1]) == true_labels_1).sum())
            num_tom_obs += len(true_labels_1)
            print(type(num_tom_correct))
            print(type(num_tom_obs))
            print("acc = ", num_tom_correct/num_tom_obs)"""

            output_crf = torch.zeros(outputs[0].size()[0], num_of_classes * num_of_classes)
            output_crf = output_crf.to(device)
            for i in range(len(output_crf)):
                for j in range((num_of_classes) ** 2):
                    if j == 0:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][0]) * (outputs[1][i][0])
                    elif j == 1:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][0]) * (outputs[1][i][1])
                    elif j == 2:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][0]) * (outputs[1][i][2])
                    elif j == 3:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][1]) * (outputs[1][i][0])
                    elif j == 4:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][1]) * (outputs[1][i][1])
                    elif j == 5:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][1]) * (outputs[1][i][2])
                    elif j == 6:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][2]) * (outputs[1][i][0])
                    elif j == 7:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][2]) * (outputs[1][i][1])
                    elif j == 8:
                        output_crf[i][j] = (outputs[2][i][j]) * (outputs[0][i][2]) * (outputs[1][i][2])
            softmax = nn.Softmax(dim=1)
            output_crf = softmax(output_crf)
            predicted_prob, predicted = torch.max(output_crf.data, 1)
            #print(predicted)

            for i in range(len(true_labels_1)):
                num_of_obs += 1
                if (true_labels_1[i] == 0) and (predicted[i] in (0, 1, 2)):
                    num_of_correct_1 += 1
                elif (true_labels_1[i] == 1) and (predicted[i] in (3, 4, 5)):
                    num_of_correct_1 += 1
                elif (true_labels_1[i] == 2) and (predicted[i] in (6, 7, 8)):
                    num_of_correct_1 += 1
            for i in range(len(true_labels_2)):
                if (true_labels_2[i] == 0) and (predicted[i] in (0, 3, 6)):
                    num_of_correct_2 += 1
                elif (true_labels_2[i] == 1) and (predicted[i] in (1, 4, 7)):
                    num_of_correct_2 += 1
                elif (true_labels_2[i] == 2) and (predicted[i] in (2, 5, 8)):
                    num_of_correct_2 += 1


    print("class_1 ACC:", num_of_correct_1/num_of_obs)
    print("class_2 ACC:", num_of_correct_2 / num_of_obs)


if __name__ == '__main__':

    # config.yaml control the settings and other hyper parameters
    config = yaml.safe_load(open("config.yaml"))
    num_of_classes = config["MLP"]["num_of_classes"]
    print("\nBuilding the train dataset:")
    train = StockExchangeDataset(config["Data"]["train_set"])
    print("\nBuilding the test dataset:")
    # the test use the id to idx of the train
    test = StockExchangeDataset(config["Data"]["test_set"], train.id_to_idx)

    class_1_classifier = MainClassifier('class_1', 'conv', train, test, num_of_classes)
    class_2_classifier = MainClassifier('class_2', 'conv', train, test, num_of_classes)
    class_3_classifier = MainClassifier('class_3', 'conv', train, test, num_of_classes*num_of_classes)

    # train_all(class_1_classifier, class_2_classifier, class_3_classifier)

    load_all(class_1_classifier, class_2_classifier, class_3_classifier)

    test_all_together(class_1_classifier, class_2_classifier, class_3_classifier)