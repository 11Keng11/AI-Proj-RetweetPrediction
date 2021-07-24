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
from utils import getWordEmbeddings

class TweetsCOV19Dataset(Dataset):
    def __init__(self, mode = "train", forClassifier=False):
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be 'train', 'val' or 'test'.")
        self._mode = mode
        self.forClassifier = forClassifier
        # hardcoded total number of rows based on self._mode
        if self._mode == "train":
            self.totalRows = 13978691
        elif self._mode == "val":
            self.totalRows = 2995431
        elif self._mode == "test":
            self.totalRows = 2995431
        with tqdm(total=self.totalRows, desc="Loading {} data".format(mode)) as bar:
            self.csv_file = pd.read_csv(self.get_dataset_path(self._mode), skiprows=lambda x: bar.update(1) and False, index_col=False, dtype=np.float32)

        self.preprocess()
        # self.preprocess_normalized()

        # clear up memory
        self.csv_file = None


    # def preprocess_normalized(self):
    #     '''Preprocess normalization function for self.csv_file'''
    #     print ("Processing {} data".format(self._mode))
    #     # remove the problem features
    #     self.csv_file.drop(columns=["Tweet ID"], inplace=True)
    #
    #     def getWeekday(values):
    #         data = []
    #         for value in tqdm(values):
    #             weekday = datetime.fromtimestamp(value).strftime("%A")
    #             weekdayEnum = {"Monday":0,
    #                            "Tuesday":1,
    #                            "Wednesday":2,
    #                            "Thursday":3,
    #                            "Friday":4,
    #                            "Saturday":5,
    #                            "Sunday":6}
    #             data.append(weekdayEnum[weekday])
    #         return pd.Series(weekdayEnum[weekday])
    #
    #     # change time to weekday
    #     self.csv_file["Timestamp"] = getWeekday(self.csv_file["Timestamp"].values)
    #     # self.csv_file["Timestamp"] = self.csv_file["Timestamp"].apply(getWeekday)
    #     print ("Timestamp converted to weekday")
    #
    #     # create a normalized_df here
    #     normalized_df = (self.csv_file-self.csv_file.min())/(self.csv_file.max()-self.csv_file.min())
    #     print ("Normalized dataframe")
    #
    #     # X_data = normalized_df.loc[:, normalized_df.columns != 'No. of Retweets'].values
    #     # Y_data = self.csv_file.loc[:, self.csv_file.columns == 'No. of Retweets'].values
    #
    #     self.x_tensor = torch.tensor(normalized_df.drop(columns=["No. of Retweets"], inplace=False).values, dtype=torch.float16)
    #     self.y_tensor = torch.tensor(self.csv_file["No. of Retweets"].values, dtype=torch.float16)
    #     print ("Converted to tensor!")

    def preprocess(self):
        # check if for classifier
        def reduceToClassifier(values):
            data = []
            for value in tqdm(values, desc="Processing for Classifier"):
                value = int(value)
                if value != 0:
                    data.append(1)
                else:
                    data.append(0)

            return data

        if self.forClassifier:
            self.csv_file["No. of Retweets"] = reduceToClassifier(self.csv_file["No. of Retweets"].values)
        else:
            # we need to drop the rows that have num of retweets = 0
            self.csv_file.drop(self.csv_file.loc[self.csv_file["No. of Retweets"]==0].index, inplace=True)
            # reset the index
            self.csv_file.reset_index(inplace=True, drop=True)
            # update total rows
            self.totalRows = len(self.csv_file)
        X_data = self.csv_file.drop(columns=["No. of Retweets"], inplace=False).values
        Y_data = self.csv_file.loc[:, self.csv_file.columns == 'No. of Retweets'].values

        self.x_tensor = torch.from_numpy(X_data)
        self.y_tensor = torch.from_numpy(Y_data)



    def get_dataset_path(self, _mode):
        '''Get the path to a particular dataset file'''
        # dataset_path = os.path.join("datasets", "filtered_{}.csv".format(_mode))
        dataset_path = os.path.join("datasets", "processed_{}.csv".format(_mode))
        return dataset_path

    def __str__(self):
        '''Print description about this dataset'''
        _str = "This dataset is a {} dataset\n".format(self._mode)
        _str += "It contains {} data samples\n".format(len(self))
        return _str

    def __len__(self):
        '''Return count of dataset'''
        # return len(self.csv_file)
        return self.totalRows

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]


def get_data_loader(mode="train", batch_size=64, forClassifier=False):
    '''Get the Dataloader, mode can be train or validation'''
    dataset = TweetsCOV19Dataset(mode=mode, forClassifier=forClassifier)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), dataset.x_tensor.shape[1] #num workers > 0 & pin_memory = True means dataloading will be async

if __name__ == "__main__":
    loader, inputSize = get_data_loader("test", 1)
    print ("Input Size: {}".format(int(inputSize)))
    x, y = next(iter(loader))
    print (x)
    print (y)
