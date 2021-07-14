'''
Utils.py holds functions / modules required for more than 1 script
'''
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class RMSLELoss(nn.Module):
    '''Root Mean Square Logarithmic Error is a custom loss function
    Adapted from: https://stackoverflow.com/questions/67648033/mean-squared-logarithmic-error-using-pytorch
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred+1), torch.log(actual+1)))

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


if __name__ == "__main__":
    embeddings = getWordEmbeddings()
    print (embeddings["florida"], type(embeddings["florida"]))
