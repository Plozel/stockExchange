import yaml
import csv

from timeit import default_timer as timer
from classifiers import MainClassifier

if __name__ == '__main__':
    # config.yaml control the settings and other hyper parameters
    config = yaml.safe_load(open("config.yaml"))

    # class_1_classifier = MainClassifier('class_1')
    class_2_classifier = MainClassifier('class_2', 'mlp')

    time_id, max_test_acc, epoch_test_max, directory_path, start_time = class_2_classifier.run_train()

    with open('../TrainedModels/general_log.csv'.format(directory_path, time_id), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [time_id, max_test_acc, epoch_test_max, *config["MLP"]])

    end_time = timer()
    print("Finish training, it took: {} sec ".format(round(end_time - start_time, 2)))







