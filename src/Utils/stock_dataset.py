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

        # normalize the features
        # features = (features - features.mean()) / features.std()

        # change "25/75/YE" from categorical to integer
        self.data["25/75/YE"], _ = pd.factorize( self.data["25/75/YE"])

        # scale the features between 0 to 1
        num_rows = features.shape[0]
        if self.set_name == "train_class_1_2_new_new":

            s_features = []
            for i in range(50):
                current_slice = features.iloc[int(i * (num_rows/50)):int((i+1) * (num_rows/50)), :]
                current_slice = (current_slice - current_slice.mean()) / current_slice.std()
                current_slice = (current_slice - current_slice.min()) / current_slice.max()
                s_features.append(current_slice)
            features = pd.concat(s_features)
        else:
            # features = (features - features.mean()) / features.std()
            # features = (features - features.min()) / features.max()
            s_features = []
            for i in range(10):
                current_slice = features.iloc[int(i * (num_rows/10)):int((i+1) * (num_rows/10)), :]
                current_slice = (current_slice - current_slice.mean()) / current_slice.std()
                current_slice = (current_slice - current_slice.min()) / current_slice.max()
                s_features.append(current_slice)
            features = pd.concat(s_features)

        labels_1 = self.data.loc[:, 'class_1']
        labels_2 = self.data.loc[:, 'class_2']
        labels_3 = self.data.loc[:, 'class_3']

        # TODO: ugly...
        # in case of old labels
        # # if self.set_name == "train_class":
        # labels = labels.replace({1: 3, 2: 3, 7: 5, 6: 5})
        # labels = labels.replace({3: 0, 4: 1, 5: 2})

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

#
# class StockExchangeSeqDataset(Dataset):
#     """Creates a Dataset object for the StockExchange data."""
#
#     def __init__(self, set_name, id_to_idx=None, class_name):
#         super().__init__()
#         self.input_size = config["MLP"]["input_size"]
#         self.target_idx = config["MLP"]["target_idx"]
#         self.set_name = set_name
#         self.data = pd.read_csv("../data/{}.csv".format(set_name), quoting=csv.QUOTE_NONE, error_bad_lines=False)
#         self.data = self.data.sort_values(by=['SH ID', 'Date'])
#         # fill missing values
#         self.data = self.data.where(pd.notna(self.data), self.data.mean(), axis='columns')
#         self.id_to_idx = {}
#         if id_to_idx:
#             self.id_to_idx = id_to_idx
#         else:
#             self.create_id_dict()
#
#         self.dataset = self.preprocess_data()
#
#     def create_id_dict(self):
#         stocks_id = self.data.iloc[:, 0]
#         stocks_idx = pd.unique(stocks_id.values.ravel())
#         stocks_idx = pd.Series(np.arange(len(stocks_idx)), stocks_idx)
#         self.id_to_idx = stocks_idx.to_dict()
#
#     def preprocess_data(self):
#         """Clean, handle missing data and covert the data to a TensorDataset object"""
#         stocks_id = self.data.iloc[:, 0].copy()
#         stocks_id = stocks_id.map(self.id_to_idx)
#         features = self.data.iloc[:, np.r_[25:79, 12:16]]
#
#         # change "25/75/YE" from categorical to integer
#         self.data["25/75/YE"], _ = pd.factorize(self.data["25/75/YE"])
#
#         num_rows = features.shape[0]
#         if self.set_name == "train_class_1_2_new_new":
#
#             s_features = []
#             for i in range(10):
#                 current_slice = features.iloc[int(i * (num_rows/10)):int((i+1) * (num_rows/10)), :]
#                 # normalize the features
#                 current_slice = (current_slice - current_slice.mean()) / current_slice.std()
#                 # scale the features between 0 to 1
#                 current_slice = (current_slice - current_slice.min()) / current_slice.max()
#                 s_features.append(current_slice)
#             features = pd.concat(s_features)
#         else:
#             features = (features - features.mean()) / features.std()
#             features = (features - features.min()) / features.max()
#
#         labels = self.data.loc[:, 'class_1']
#         # TODO: ugly...
#         # in case of old labels
#         labels = labels.replace({1: 3, 2: 3, 7: 5, 6: 5})
#         labels = labels.replace({3: 0, 4: 1, 5: 2})
#
#         print("{} distribution:".format(self.set_name))
#         print(labels.value_counts(normalize=True))
#         df = pd.concat([stocks_id, self.data["25/75/YE"].copy(), features, labels], axis=1)
#         grouped = df.groupby(df["SH ID"])
#         sml_seq = []
#         stocks_id_seq = []
#         features_seq = []
#         real_y_1_seq = []
#         lengths = []
#         stocks_id = stocks_id.dropna()
#         for idx in pd.unique(stocks_id):
#
#             grouped_idx = grouped.get_group(idx)
#             group_len = len(grouped_idx.values.tolist())
#             sml_seq.append(grouped_idx["25/75/YE"].values.tolist())
#             stocks_id_seq.append(grouped_idx["SH ID"].values.tolist())
#             features_seq.append(grouped_idx.iloc[:, np.r_[36:79, 11:26]].values.tolist())
#             real_y_1_seq.append(grouped_idx.loc[:, 'class_1'].values.tolist())
#             lengths.append(group_len)
#
#
#         # Padding
#         max_len = max([len(batch) for batch in sml_seq])
#         sml_seq = [batch + [-1] * (max_len - len(batch)) for batch in sml_seq]
#         stocks_id_seq = [batch + [-1] * (max_len - len(batch)) for batch in stocks_id_seq]
#         features_seq = [batch + [[0]*len(batch[0])] * (max_len - len(batch)) for batch in features_seq]
#         real_y_1_seq = [batch + [0] * (max_len - len(batch)) for batch in real_y_1_seq]
#         # lengths = [batch + [-1] * (max_len - len(batch)) for batch in lengths]
#
#         sml = torch.tensor(sml_seq).long()
#         stocks_id = torch.tensor(stocks_id_seq).long()
#
#         features = torch.FloatTensor(features_seq)
#         real_y_1 = torch.tensor(real_y_1_seq).long()
#         lengths = torch.tensor(lengths).long()
#         dataset = TensorDataset(stocks_id, sml, features, real_y_1, lengths)
#         return dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         return self.dataset[index]
