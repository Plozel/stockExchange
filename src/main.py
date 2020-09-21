import yaml
import csv

from Utils.stock_dataset import StockExchangeDataset

from classifiers import MainClassifier

if __name__ == '__main__':
    # config.yaml control the settings and other hyper parameters
    config = yaml.safe_load(open("config.yaml"))

    print("\nBuilding the train dataset:")
    train = StockExchangeDataset(config["Data"]["train_set"])
    print("\nBuilding the test dataset:")
    # the test use the id to idx of the train
    test = StockExchangeDataset(config["Data"]["test_set"], train.id_to_idx)

    class_1_classifier = MainClassifier('class_1', 'mlp', train, test)
    class_2_classifier = MainClassifier('class_2', 'conv', train, test)

    print("Trains on class_1")
    print("--------------------")
    time_id, max_test_acc, epoch_test_max = class_1_classifier.run_train()

    with open('../TrainedModels/general_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, max_test_acc, epoch_test_max, *config["MLP"]])

    print("Trains on class_2")
    print("--------------------")
    time_id, max_test_acc, epoch_test_max = class_2_classifier.run_train()

    with open('../TrainedModels/general_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, max_test_acc, epoch_test_max, *config["MLP"]])









