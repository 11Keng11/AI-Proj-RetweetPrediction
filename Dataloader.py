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

        self.preprocess()
        # self.preprocess_normalized()

    def preprocess_normalized(self):
        '''Preprocess normalization function for self.csv_file'''
        # remove the 2 problem features
        self.csv_file = self.csv_file[self.csv_file.columns[~self.csv_file.columns.isin(["Tweet ID", "Timestamp"])]]

        # create a normalized_df here
        normalized_df = (self.csv_file-self.csv_file.min())/(self.csv_file.max()-self.csv_file.min())

        X_data = normalized_df.loc[:, normalized_df.columns != 'No. of Retweets'].values
        Y_data = self.csv_file.loc[:, self.csv_file.columns == 'No. of Retweets'].values

        self.x_tensor = torch.tensor(X_data, dtype=torch.float32)
        self.y_tensor = torch.tensor(Y_data, dtype=torch.float32)

    def preprocess(self):
        embeddings = getWordEmbeddings()
        def entityEmbed(embeddings, value):
            embed = np.zeros((1, 50)).astype(np.float32)
            if value != "null;":
                entities = value.split(":")
                count = 0
                for entity in entities:
                    words = entity.split(" ")
                    for word in words:
                        count += 1
                        if word in embeddings:
                            embed += embeddings[word]
                        else:
                            embed += np.random.rand(1,50).astype(np.float32)
                embed /= count
            return embed.flatten()

        print ("Generating Entity Embeddings.... This may take a while")
        self.csv_file["Entities"] = self.csv_file["Entities"].apply(lambda x: entityEmbed(embeddings, x), 1)
        del embeddings
        print (self.csv_file.loc[0, :].values)
        X_data = self.csv_file.drop(columns=["No. of Retweets", "Entities"], inplace=False).values
        # X_data = self.csv_file.loc[:, self.csv_file.columns != 'No. of Retweets'].values
        Y_data = self.csv_file.loc[:, self.csv_file.columns == 'No. of Retweets'].values
        self.x_tensor = torch.tensor(X_data, dtype=torch.float32)
        self.y_tensor = torch.tensor(Y_data, dtype=torch.float32)

        # add the embeddings as last 50 dimensions
        add = self.csv_file["Entities"]
        addTensor = torch.tensor(add, dtype=torch.float32)
        print (self.x_tensor.shape)
        print (addTensor.shape)
        self.x_tensor = torch.cat((self.x_tensor, addTensor), 1)

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
        return len(self.csv_file)

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]


def get_data_loader(mode="train", batch_size=64):
    '''Get the Dataloader, mode can be train or validation'''
    dataset = TweetsCOV19Dataset(mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True) #num workers > 0 & pin_memory = True means dataloading will be async

if __name__ == "__main__":
    loader = get_data_loader("test", 1)
    x, y = next(iter(loader))
    print (x.shape)
    print (y.shape)
