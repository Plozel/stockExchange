import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
import csv
import yaml

config = yaml.safe_load(open("config.yaml"))


class StockExchangeDataset(Dataset):
    """Creates a Dataset object for the StockExchange data."""

    def __init__(self, set_name):
        super().__init__()
        self.input_size = config["MLP"]["input_size"]
        self.target_idx = config["MLP"]["target_idx"]

        self.data = pd.read_csv("../data/{}.csv".format(set_name), quoting=csv.QUOTE_NONE, error_bad_lines=False)
        self.dataset = self.preprocess_data()


    def preprocess_data(self):
        """Clean, handle missing data and covert the data to a TensorDataset object"""

        # fill missing values
        self.data.where(pd.notna(self.data), self.data.mean(), axis='columns')
        # features = self.data.iloc[:, (self.target_idx - 1 - self.input_size):self.target_idx - 1]
        features = self.data.iloc[:, np.r_[30:79, 12:17]]
        # normalize the features
        features = (features - features.mean()) / features.std()

        # change "25/75/YE" from categorical to integer # TODO add embedding
        # features["25/75/YE"], _ = pd.factorize(features["25/75/YE"])
        # scale the features between 0 to 1
        features = (features - features.min()) / features.max()

        labels = self.data.loc[:, 'class']
        # TODO: ugly...
        labels = labels.replace({1: 3, 2: 3, 7: 5, 6: 5})
        labels = labels.replace({3: 0, 4: 1, 5: 2})

        print(labels.value_counts(normalize=True))
        features = torch.tensor(features.values).float()
        real_y_1 = torch.tensor(labels.values).long()
        real_y_2 = torch.tensor(self.data.loc[:, 'class'].values).long()
        dataset = TensorDataset(features, real_y_1, real_y_2)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


