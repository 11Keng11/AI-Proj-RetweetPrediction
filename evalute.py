'''
#Retweets Evaluation
with reference to: https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.14.0113.0657
'''
import os
import torch
import argparse
# import seaborn as sns
# from model import PCRNN
from Dataloader import get_data_loader
from utils import RMSLELoss # custom loss function
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import  accuracy_score, f1_score, recall_score, precision_score

#========== ARGPARSE BLOCK ==========#
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model file to be used for evaluation", type=str, required=True)

def parserSummary(args):
    print ("Evaluating #Retweets Model: {}.".format(args.model))
#======= END OF ARGPARSE BLOCK =======#

def runTestOnModel(modelPath):
    '''Run model on test set and
    return a list of predictions, truths and test loss
    '''
    assert modelPath, "No Model Path provided"
    # model = MODEL()
    model.load_state_dict(torch.load(modelPath))
    testLoader = get_data_loader(mode="test", batch_size=1)

    model.eval()
    predictions = []
    truths = []
    testLoss = 0
    criterion = RMSLELoss()
    for input_x, target in tqdm(testLoader, desc="Testing..."):
        output = model(input_x)
        actuals.append(target.detach().numpy())
        predictions.append(output.detach().numpy())
        # calculate loss
        testLoss += criterion(output, target)

    return predictions, truths, testLoss


def evaluate(modelPath):
    '''Evaluate the given model with test data'''
    pass
    preds, truths, testLoss = runTestOnModel(modelPath)
    # Scores
    f1 = f1_score(truths, preds, average="weighted")
    recall = recall_score(truths, preds, average="weighted")
    precision = precision_score(truths, preds, average="weighted")
    accuracy = accuracy_score(truths, preds)
    print("F1: {:6f}".format(f1))
    print("Recall: {:6f}".format(recall))
    print("Precision: {:6f}".format(precision))
    print("Accuracy: {:6f}".format(accuracy))

if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    evaluate(args.model)
