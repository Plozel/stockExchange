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
        self.input_size = config["MLP"]["input_size"]
        self.target_idx = config["MLP"]["target_idx"]
        self.set_name = set_name
        self.data = pd.read_csv("../data/{}.csv".format(set_name), quoting=csv.QUOTE_NONE, error_bad_lines=False)
        self.id_to_idx = {}
        if id_to_idx:
            self.id_to_idx = id_to_idx
        else:
            self.create_id_dict()

        self.dataset = self.preprocess_data()

    def create_id_dict(self):
        stocks_id = self.data.iloc[:, 0]
        stocks_idx = self.data.iloc[:, 0].copy()
        stocks_idx = pd.unique(stocks_id.values.ravel())
        stocks_idx = pd.Series(np.arange(len(stocks_idx)), stocks_idx)
        self.id_to_idx = stocks_idx.to_dict()

    def preprocess_data(self):
        """Clean, handle missing data and covert the data to a TensorDataset object"""
        # fill missing values
        # self.data = self.data.where(pd.notna(self.data), self.data.mean(), axis='columns')
        stocks_id = self.data.iloc[:, 0]
        stocks_id = stocks_id.map(self.id_to_idx)

        features = self.data.iloc[:, np.r_[36:79, 11:26]]
        print(features.columns)
        # normalize the features
        # features = (features - features.mean()) / features.std()

        # change "25/75/YE" from categorical to integer
        self.data["25/75/YE"], _ = pd.factorize( self.data["25/75/YE"])

        # scale the features between 0 to 1
        num_rows = features.shape[0]
        if self.set_name == "train_class":

            s_features = []
            for i in range(10):
                current_slice = features.iloc[int(i * (num_rows/10)):int((i+1) * (num_rows/10)), :]
                current_slice = (current_slice - current_slice.mean()) / current_slice.std()
                current_slice = (current_slice - current_slice.min()) / current_slice.max()
                s_features.append(current_slice)
            features = pd.concat(s_features)
        else:
            features = (features - features.mean()) / features.std()
            features = (features - features.min()) / features.max()

        labels = self.data.loc[:, 'class_1']
        labels_2 = self.data.loc[:, 'class_2']
        real_y_1_2 = torch.zeros([len(labels)], dtype=torch.long)
        for i in range(len(labels)):
            if labels[i] == 0:
                if labels_2[i] == 0:
                    real_y_1_2[i] = 0
                elif labels_2[i] == 1:
                    real_y_1_2[i] = 1
                else:
                    real_y_1_2[i] = 2
                continue
            elif labels[i] == 1:
                if labels_2[i] == 0:
                    real_y_1_2[i] = 3
                elif labels_2[i] == 1:
                    real_y_1_2[i] = 4
                else:
                    real_y_1_2[i] = 5
                continue
            else:
                if labels_2[i] == 0:
                    real_y_1_2[i] = 6
                elif labels_2[i] == 1:
                    real_y_1_2[i] = 7
                else:
                    real_y_1_2[i] = 8

        print('#######################################################')
        print(labels)
        print('#######################################################')


        print("{} labels_1 distribution:".format(self.set_name))
        print(labels.value_counts(normalize=True))


        sml = torch.tensor(self.data["25/75/YE"].values).long()
        stocks_id = torch.tensor(stocks_id.values).long()
        features = torch.tensor(features.values).float()
        real_y_1 = torch.tensor(labels.values).long()
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(real_y_1)
        print(real_y_1.shape)
        print(real_y_1_2.shape)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        real_y_2 = torch.tensor(labels_2.values).long()
        dataset = TensorDataset(stocks_id, sml, features, real_y_1, real_y_2, real_y_1_2)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


