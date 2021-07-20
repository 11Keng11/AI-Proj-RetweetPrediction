'''
#Retweets Evaluation
with reference to: https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.14.0113.0657
'''
import os
import torch
import argparse
from glob import glob
import re
from Regression_NN_1 import Net
from Dataloader import get_data_loader
from utils import *
from tqdm import tqdm
import plotly.express as px

#========== ARGPARSE BLOCK ==========#
def checkIfModelNameExists(parser, arg):
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, arg)
    if os.path.exists(saveDir):
        return arg
    else:
        raise argparse.ArgumentTypeError("No such model name: {} exists!".format(arg))
        return

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Training Name to load in data on", type=lambda x: checkIfModelNameExists(parser, x), required=True)
parser.add_argument("-b", "--batch", help="Test batch size", type=int, default=8192)
# parser.add_argument("-m", "--model", help="model file to be used for evaluation", type=str, required=True)

def parserSummary(args):
    print ("Evaluating #Retweets Model from model name: {}.".format(args.name))
    print ("Batch Size: {}".format(args.batch))
#======= END OF ARGPARSE BLOCK =======#

def runTestOnModel(checkpoint, batch_size):
    '''Run model on test set and conduct plots on the training stats
    '''
    # get test loader
    testLoader, inputSize = get_data_loader(mode="test", batch_size=batch_size)

    model = Net(inputSize)
    model.load_state_dict(checkpoint)

    # move model to gpu
    model = model.to("cuda")

    # set the model to eval
    model.eval()
    testLoss = 0
    criterion = MSLELoss()
    with tqdm(testLoader, desc="Testing", leave=False) as testBar:
        for input_x, target in testBar:
            # Move to gpu
            input_x = input_x.to("cuda")
            target = target.to("cuda")
            output = model(input_x)
            # calculate loss
            loss = criterion(output, target)
            testLoss += loss.cpu().item()
            testBar.set_postfix(loss = testLoss)
    # average the test loss
    testLoss /=  int(len(testLoader.dataset)/batch_size)
    return testLoss

def plotTrainingStats(modelName, testLoss):
    # get training stats
    df = getTrainingStats(modelName)
    fig = px.scatter(df, x="epoch", y=["train loss", "val loss"])
    # update axis, titles
    fig.update_layout(
                        title="{} Training Stats".format(modelName),
                        xaxis_title="Num Epoch",
                        yaxis_title="Mean Squared Log Error",
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="RebeccaPurple"
                        )
                    )
    fig.add_hline(y=testLoss, line_width=3, line_dash="dash", line_color="green", annotation_text="Test Loss")
    fig.show()

def evaluate(modelName, batch_size):
    '''Evaluate the given model with test data'''
    # load in model checkpoint
    checkpoint = getBestModel(modelName)
    # run test on model
    testLoss = runTestOnModel(checkpoint, batch_size)
    # Scores
    print ("Test Loss: {}".format(testLoss))
    # plot training stats
    plotTrainingStats(modelName, testLoss)

if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    evaluate(args.name, args.batch)
