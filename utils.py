'''
Utils.py holds functions / modules required for more than 1 script
'''
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from glob import glob
import re
import os
import pandas as pd

class RMSLELoss(nn.Module):
    '''Root Mean Square Logarithmic Error is a custom loss function
    Adapted from: https://stackoverflow.com/questions/67648033/mean-squared-logarithmic-error-using-pytorch
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        # print ("PRED: ", pred)
        # print ("min: ", torch.)
        # print ("ACTUAL: ", actual)
        # print ("MSE: ", self.mse(torch.log(pred+1), torch.log(actual+1)))
        return torch.sqrt(self.mse(torch.log(pred+1+1e-6), torch.log(actual+1)) + 1e-6)

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred+1+1e-6), torch.log(actual+1))

def getWordEmbeddings():
    '''function uses pre-trained word embeddings to return a dictionary'''
    embeddings = {}
    with open("glove.6B.50d.txt", "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Getting Word Embeddings"):
            splitted = line.strip().split(" ")
            word = splitted[0]
            embeds = splitted[1:]
            embeddings[word] = np.array(embeds, dtype="float32")

    return embeddings

def getModelCheckpoint(modelName):
    '''Function returns latest model checkpoint'''
    modelFolder = "Models"
    modelSavesFolder = "ModelSaves"
    saveDir = os.path.join(modelFolder, modelName, modelSavesFolder)
    modelSaves = glob(saveDir+"/*.pth")

    modelSaves.sort(key=lambda f: int(re.sub("\D", "", f)))
    print ("Loading Checkpoint: {}".format(modelSaves[-1]))
    return torch.load(modelSaves[-1])

def getBestModel(modelName):
    '''Function returns best model.pth'''
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName, "BestModel.pth")
    return torch.load(saveDir)

def getTrainingStats(modelName):
    '''Function returns model training stats as a df'''
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName)
    df = pd.read_csv(os.path.join(saveDir, "trainingStats.csv"))
    return df


if __name__ == "__main__":
    embeddings = getWordEmbeddings()
    print (embeddings["cereal"])
