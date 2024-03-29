import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
import csv
import yaml

config = yaml.safe_load(open("config.yaml"))


class StockExchangeDataset(Dataset):
    """Creates a Dataset object for the StockExchange data."""

    def __init__(self, set_name, id_to_idx=None):
        super().__init__()
        self.input_size = config["CommonParameters"]["input_size"]
        self.target_idx = config["CommonParameters"]["target_idx"]
        self.set_name = set_name
        self.data = pd.read_csv("../data/{}.csv".format(set_name), quoting=csv.QUOTE_NONE, error_bad_lines=False)
        self.id_to_idx = {}
        if id_to_idx:
            self.id_to_idx = id_to_idx
        else:
            self.create_id_dict()

        self.dataset = self.preprocess_data()

    def create_id_dict(self):
        """Creates a stock_id-idx dictionary."""

        stocks_id = self.data.iloc[:, 0]
        stocks_idx = self.data.iloc[:, 0].copy()
        stocks_idx = pd.unique(stocks_id.values.ravel())
        stocks_idx = pd.Series(np.arange(len(stocks_idx)), stocks_idx)
        self.id_to_idx = stocks_idx.to_dict()

    def preprocess_data(self):
        """Clean, scale, normalize, handle missing data and convert the data to a TensorDataset object"""

        # fill missing values
        self.data = self.data.where(pd.notna(self.data), self.data.mean(), axis='columns')
        stocks_id = self.data.iloc[:, 0]
        stocks_id = stocks_id.map(self.id_to_idx)

        features = self.data.iloc[:, np.r_[11:79]]

        # change "25/75/YE" from categorical to integer
        self.data["25/75/YE"], _ = pd.factorize(self.data["25/75/YE"])

        # scale the features between 0 to 1
        num_rows = features.shape[0]
        train_slices = config["CommonParameters"]["train_num_of_slices"]
        test_slices = config["CommonParameters"]["test_num_of_slices"]
        if self.set_name == config["Data"]["train_set"]:

            s_features = []
            for i in range(train_slices):
                current_slice = features.iloc[int(i * (num_rows/train_slices)):int((i+1) * (num_rows/train_slices)), :]
                current_slice = (current_slice - current_slice.mean()) / current_slice.std()
                current_slice = (current_slice - current_slice.min()) / current_slice.max()
                s_features.append(current_slice)
            features = pd.concat(s_features)
        else:
            s_features = []
            for i in range(test_slices):
                current_slice = features.iloc[int(i * (num_rows/test_slices)):int((i+1) * (num_rows/test_slices)), :]
                current_slice = (current_slice - current_slice.mean()) / current_slice.std()
                current_slice = (current_slice - current_slice.min()) / current_slice.max()
                s_features.append(current_slice)
            features = pd.concat(s_features)

        labels_1 = self.data.loc[:, 'class_1']
        labels_2 = self.data.loc[:, 'class_2']
        labels_3 = self.data.loc[:, 'class_3']

        print("{} class_1 distribution:".format(self.set_name))
        print(labels_1.value_counts(normalize=True))
        print(" ")
        print("{} class_2 distribution:".format(self.set_name))
        print(labels_2.value_counts(normalize=True))
        print(" ")
        print("{} class_3 distribution:".format(self.set_name))
        print(labels_3.value_counts(normalize=True))

        sml = torch.tensor(self.data["25/75/YE"].values).long()
        stocks_id = torch.tensor(stocks_id.values).long()
        features = torch.tensor(features.values).float()
        real_y_1 = torch.tensor(labels_1.values).long()
        real_y_2 = torch.tensor(labels_2.values).long()
        real_y_3 = torch.tensor(labels_3.values).long()

        dataset = TensorDataset(stocks_id, sml, features, real_y_1, real_y_2, real_y_3)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
