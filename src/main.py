import yaml
import csv

from Utils.stock_dataset import StockExchangeDataset

from classifiers import MainClassifier


def run(classifier):

    print("Trains on {}".format(classifier))
    print("--------------------")
    time_id, max_test_acc, epoch_test_max = classifier.run_train()

    with open('../TrainedModels/general_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, max_test_acc, epoch_test_max, *config["MLP"]])


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
    class_2_classifier = MainClassifier('class_3', 'conv', train, test, num_of_classes)
    class_3_classifier = MainClassifier('class_3', 'conv', train, test, num_of_classes*num_of_classes)

    run(class_3_classifier)

