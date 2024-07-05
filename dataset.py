import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Custom_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        features = torch.tensor(self.dataframe.iloc[index, :-1])
        label =  torch.tensor(self.dataframe.iloc[index, -1:])
        return features, label