from Utils import stock_dataset
from Models.LSTM import LSTMModel
from torch.utils.data import dataloader

dataset = stock_dataset.StockExchangeDataset()
model = LSTMModel(78, 10, 2)


