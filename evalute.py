'''
#Retweets Evaluation
with reference to: https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.14.0113.0657
'''
import os
import torch
import argparse
from glob import glob
import re
from Regression_NN_1 import *
from Dataloader import get_data_loader
from utils import *
from tqdm import tqdm
import plotly.express as px

#========== ARGPARSE BLOCK ==========#
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

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
parser.add_argument("-c", "--classifier", type=str_to_bool, default=False)

def parserSummary(args):
    print ("Evaluating Model: {}.".format(args.name))
    print ("Batch Size: {}".format(args.batch))
    print ("For classifier: {}".format(args.classifier))
#======= END OF ARGPARSE BLOCK =======#

def runTestOnModel(checkpoint, batch_size, forClassifier=False):
    '''Run model on test set and conduct plots on the training stats
    '''
    # get test loader
    testLoader, inputSize = get_data_loader(mode="test", batch_size=batch_size, forClassifier=forClassifier)

    if forClassifier:
        model = Binary_Classifier(inputSize)
    else:
        model = Net2(inputSize)

    model.load_state_dict(checkpoint)

    # move model to gpu
    model = model.to("cuda")

    # set the model to eval
    model.eval()
    testLoss = 0

    if forClassifier:
        criterion = nn.BCELoss()
    else:
        criterion = MSLELoss()

    accuracy = 0
    targets = []
    preds = []
    with tqdm(testLoader, desc="Testing", leave=False) as testBar:
        for input_x, target in testBar:
            # Move to gpu
            input_x = input_x.to("cuda")
            target = target.to("cuda")
            output = model(input_x.float())
            # calculate loss
            loss = criterion(output, target.float())
            testLoss += loss.cpu().item() * len(target)
            testBar.set_postfix(loss = testLoss)

            accuracy += (output.cpu().detach().numpy().round().astype(np.int) == target.cpu().numpy().round().astype(np.int)).mean()

            preds.extend(output.cpu().detach().numpy().round().astype(np.int).flatten().tolist())
            targets.extend(target.cpu().numpy().astype(np.int).flatten().tolist())

    # average the test loss
    testLoss /=  len(testLoader.dataset)
    accuracy /= int(len(testLoader.dataset)/batch_size)

    print ("Accuracy: {}".format(accuracy))

    print ("Preds: {}".format(preds[:10]))
    print ("Targets: {}".format(targets[:10]))

    return testLoss, preds, targets

def plotTrainingStats(modelName, testLoss, forClassifier=False):
    # get training stats
    df = getTrainingStats(modelName)
    fig = px.scatter(df, x="epoch", y=["train loss", "val loss"])
    fig.add_hline(y=testLoss, line_width=3, line_dash="dash", line_color="green", annotation_text="Best Model Test Loss")
    # update axis, titles
    if forClassifier:
        yAxis = "BCELoss"
    else:
        yAxis = "Mean Squared Log Error (MSLE)"
    fig.update_layout(
                        title="{} Training Stats".format(modelName),
                        xaxis_title="Num Epoch",
                        yaxis_title=yAxis,
                        font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="RebeccaPurple"
                        )
                    )

    fig.show()

def evaluate(modelName, batch_size, forClassifier=False):
    '''Evaluate the given model with test data'''
    # load in model checkpoint
    checkpoint = getBestModel(modelName)
    # run test on model
    testLoss, preds, targets = runTestOnModel(checkpoint, batch_size, forClassifier)
    # save to file
    savePredToFile(preds, targets, modelName)
    # Scores
    print ("Test Loss: {}".format(testLoss))
    # plot training stats
    plotTrainingStats(modelName, testLoss, forClassifier)


def savePredToFile(preds, targets, modelName):
    modelFolder = "Models"
    savePath = os.path.join(modelFolder, modelName, "Preds.csv")
    with open(savePath, "w") as f:
        f.write("Pred, Target \n")
        for index, pred in tqdm(enumerate(preds), total = len(preds), desc="Saving Preds to {}".format(savePath)):
            f.write("{},{}\n".format(pred, targets[index]))

if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    evaluate(args.name, args.batch, args.classifier)
