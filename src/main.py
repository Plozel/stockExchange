import yaml
import csv
import numpy as np
import torch
from tqdm import tqdm

from Utils.stock_dataset import StockExchangeDataset
from classifiers import MainClassifier


def run_training(classifier):
    """Runs the training phase for the requested classifier."""

    print("Trains to classify {}".format(classifier.target_name))
    print("--------------------")
    time_id, max_test_acc, epoch_test_max = classifier.run_train()

    with open('../TrainedModels/general_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, classifier.target_name, max_test_acc, epoch_test_max, *config["CommonParameters"]])


def train_all(_class_1_classifier, _class_2_classifier, _class_3_classifier):
    """Trains all classifiers."""

    run_training(_class_1_classifier)
    run_training(_class_2_classifier)
    run_training(_class_3_classifier)


def load_all(models_used, _class_1_classifier, _class_2_classifier, _class_3_classifier):
    """Loads all saved models"""

    print("-----------------------")
    print("Loading saved models:")
    print("-----------------------")

    _class_1_classifier.load_model(config["TrainedModels"][models_used[0]]["pkl_path_1"])
    _class_2_classifier.load_model(config["TrainedModels"][models_used[1]]["pkl_path_2"])
    _class_3_classifier.load_model(config["TrainedModels"][models_used[2]]["pkl_path_3"])


def test_all_separate(_class_1_classifier, _class_2_classifier, _class_3_classifier):
    """Tests all classifiers separately."""

    print("-----------------------")
    print("Test all models:")
    print("-----------------------")

    _class_1_classifier.run_test()
    _class_2_classifier.run_test()
    _class_3_classifier.run_test()


def mrf_inference(_class_1_classifier, _class_2_classifier, _class_3_classifier):
    """Tests all classifiers after MRF inference"""

    print("-----------------------")
    print("Test all models through mrf inference:")
    print("-----------------------")

    test_loader = _class_1_classifier.test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _class_1_classifier.model.eval()
    _class_2_classifier.model.eval()
    _class_3_classifier.model.eval()

    i = 0
    num_of_correct = [0, 0, 0]
    total = 0

    for test_batch in tqdm(test_loader):
        with torch.no_grad():
            stocks_id, sml, features, true_labels_1, true_labels_2, true_labels_3 = test_batch[0].to(device), \
                                                                                    test_batch[1].to(device), \
                                                                                    test_batch[2].to(device), \
                                                                                    test_batch[3].to(device), \
                                                                                    test_batch[4].to(device), \
                                                                                    test_batch[5].to(device)

            outputs = [_class_1_classifier.model(stocks_id, sml, features).to(device),
                       _class_2_classifier.model(stocks_id, sml, features).to(device),
                       _class_3_classifier.model(stocks_id, sml, features).to(device)]

            binary_mul_tmp = torch.einsum('ai, aj -> aij', outputs[0], outputs[1]).flatten(start_dim=1)
            crf_outputs = binary_mul_tmp * outputs[2]

            crf_outputs_1 = torch.cat(
                [crf_outputs[:, 0:3].sum(dim=1).unsqueeze(1), crf_outputs[:, 3:6].sum(dim=1).unsqueeze(1),
                 crf_outputs[:, 6:9].sum(dim=1).unsqueeze(1)], dim=1)
            crf_outputs_2 = torch.cat(
                [crf_outputs[:, np.r_[0, 3, 6]].sum(dim=1).unsqueeze(1),
                 crf_outputs[:, np.r_[1, 4, 7]].sum(dim=1).unsqueeze(1),
                 crf_outputs[:, np.r_[2, 5, 8]].sum(dim=1).unsqueeze(1)], dim=1)

            # each element in probs_and_preds is a tuple of maximum probability and
            # the prediction for a certain class- [(probs, predicted),...]
            probs_and_preds = [torch.max(crf_outputs_1.data, 1),
                               torch.max(crf_outputs_2.data, 1),
                               torch.max(crf_outputs.data, 1)]

            num_of_correct[0] += (probs_and_preds[0][1] == true_labels_1).sum()
            num_of_correct[1] += (probs_and_preds[1][1] == true_labels_2).sum()
            num_of_correct[2] += (probs_and_preds[2][1] == true_labels_3).sum()

            total += len(true_labels_1)

    for i in range(3):
        print("class_{} ACC:".format(i+1))
        print(num_of_correct[i].item()/total)


if __name__ == '__main__':

    # config.yaml control the settings and other hyper parameters
    config = yaml.safe_load(open("config.yaml"))
    num_of_classes = config["CommonParameters"]["num_of_classes"]
    print("\nBuilding the train dataset:")
    train = StockExchangeDataset(config["Data"]["train_set"])
    print("\nBuilding the test dataset:")
    # the test use the id to idx of the train
    test = StockExchangeDataset(config["Data"]["test_set"], train.id_to_idx)

    class_1_classifier = MainClassifier('class_1', 'mlp', train, test, num_of_classes)
    class_2_classifier = MainClassifier('class_2', 'mlp', train, test, num_of_classes)
    class_3_classifier = MainClassifier('class_3', 'mlp', train, test, num_of_classes*num_of_classes)

    # train_all(class_1_classifier, class_2_classifier, class_3_classifier)
    load_all(['mlp', 'mlp', 'mlp'], class_1_classifier, class_2_classifier, class_3_classifier)
    test_all_separate(class_1_classifier, class_2_classifier, class_3_classifier)
    mrf_inference(class_1_classifier, class_2_classifier, class_3_classifier)
