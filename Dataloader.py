#!/usr/bin/env python
# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TweetsCOV19Dataset(Dataset):
    def __init__(self, mode = "train"):
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be 'train', 'val' or 'test'.")
        self._mode = mode
        # hardcoded total number of rows based on self._mode
        if self._mode == "train":
            totalRows = 13978697
        elif self._mode == "val":
            totalRows = 2995436
        elif self._mode == "test":
            totalRows = 2995435
        with tqdm(total=totalRows, desc="Loading {} data".format(mode)) as bar:
            self.csv_file = pd.read_csv(self.get_dataset_path(self._mode), skiprows=lambda x: bar.update(1) and False)
        # remove the 2 problem features
        self.csv_file = self.csv_file[self.csv_file.columns[~self.csv_file.columns.isin(["Tweet ID", "Timestamp"])]]
        X_data = self.csv_file.loc[:, self.csv_file.columns != 'No. of Retweets'].values
        Y_data = self.csv_file.loc[:, self.csv_file.columns == 'No. of Retweets'].values
        self.x_tensor = torch.tensor(X_data)
        self.y_tensor = torch.tensor(Y_data)

    def get_dataset_path(self, _mode):
        '''Get the path to a particular dataset file'''
        dataset_path = os.path.join("datasets", "filtered_{}.csv".format(_mode))
        return dataset_path

    def __str__(self):
        '''Print description about this dataset'''
        _str = "This dataset is a {} dataset\n".format(self._mode)
        _str += "It contains {} data samples\n".format(len(self))
        return _str

    def __len__(self):
        '''Return count of dataset'''
        return len(self.csv_file)

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]


def get_data_loader(mode="train", batch_size=64):
    '''Get the Dataloader, mode can be train or validation'''
    dataset = TweetsCOV19Dataset(mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3) #num workers > 0 & pin_memory = True means dataloading will be async
