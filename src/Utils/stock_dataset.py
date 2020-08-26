import pandas as pd
import torch
from torch.utils.data.dataset import Dataset, TensorDataset
import csv


class StockExchangeDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("data/train_data.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False)
        self.data.where(pd.notna(self.data), self.data.mean(), axis='columns')
        self.dataset = self.convert_to_dataset()

    def convert_to_dataset(self):
        features = self.data.iloc[:, 2:80]
        # change 25/75/YE from categorical to integer
        features["25/75/YE"], _ = pd.factorize(features["25/75/YE"])
        features = torch.tensor(features.values)
        real_y_1 = torch.tensor(self.data.loc[:, 'TMRW1_IXChange'].values)
        real_y_2 = torch.tensor(self.data.loc[:, 'TMRW2_IXChange'].values)
        dataset = TensorDataset(features, real_y_1, real_y_2)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


