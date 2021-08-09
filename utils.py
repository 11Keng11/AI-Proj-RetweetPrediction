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
import calendar

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

### Data processing useful functions ###
# input String object: dateString. Example "Tue Jun 30 02:12:41 +0000 2020"
# output List object: encDateTime. Format: [day_sin, day_cos, month_sin, month_cos, time_sin, time_cos]
def encodeDateTime(dateString):
    dayStr = dateString[:3]
    monStr = dateString[4:7]
    hourStr = dateString[11:13]
    minStr = dateString[14:16]
    secStr = dateString[17:19]

    dayEnc = encodeDay(dayStr)
    monEnc = encodeMon(monStr)
    timeEnc = encodeTime(hourStr, minStr, secStr)

    encDateTime = dayEnc + monEnc + timeEnc
    return encDateTime

# input Sting obj: monStr. Example: "Jan"
# output List obj: monEnc. Format: [mon_sin, mon_cos]
def encodeMon(monStr):
    monDict = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    monInt = monDict[monStr]
    monEnc = cyclicEncode(monInt,12)
    return monEnc

# input Sting obj: dayStr. Example: "Tue"
# output List obj: monEnc. Format: [day_sin, day_cos]
def encodeDay(dayStr):
    dayDict = {day: index for index, day in enumerate(calendar.day_abbr) if day}
    dayInt = dayDict[dayStr]
    dayEnc = cyclicEncode(dayInt,7)
    return dayEnc

# input Sting obj: hourStr,minStr,secStr. Example: "02", "12", "41"
# output List obj: timeEnc. Format: [time_sin, time_cos]
def encodeTime(hourStr,minStr,secStr):
    hourInt = int(hourStr)
    secInt = int(minStr)*60 + int(secStr)
    secFlt = secInt/3600
    timeFlt = hourInt + secFlt
    timeEnc = cyclicEncode(timeFlt,24)
    return timeEnc

# input Int obj: num, size. Example: 2, 12. **2 to represent febuary. 12 to represent total number of months**
# output List obj: [x,y]. Format: [XX_sin, XX_cos]
def cyclicEncode(num, size):
    x = np.sin(2 * np.pi * num/size)
    y = np.cos(2 * np.pi * num/size)
    return [x,y]

#==== END OF DATA PROCESSING FUNCTIONS ====#


if __name__ == "__main__":
    embeddings = getWordEmbeddings()
    print (embeddings["cereal"])
