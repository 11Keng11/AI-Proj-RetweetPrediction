'''
Ensemble Model Evaluation
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
parser.add_argument("-c", "--classifier", help="Classifier Name to load in data on", type=lambda x: checkIfModelNameExists(parser, x), required=True)
parser.add_argument("-p", "--predictor", nargs='+', help="Predictor Name to load in data on", type=lambda x: checkIfModelNameExists(parser, x), required=True)
parser.add_argument("-b", "--batch", help="Test batch size", type=int, default=8192)
parser.add_argument("-d", "--data", type=str, default="processed", help="Dataset prestring")

def parserSummary(args):
    print ("Evaluating Ensemble Model.")
    print ("Using dataset pre string: {}".format(args.data))
    print ("Classifier: {}".format(args.classifier))
    print ("Predictor: {}".format(args.predictor))
    print ("Batch Size: {}".format(args.batch))
#======= END OF ARGPARSE BLOCK =======#
def runTestOnEnsemble(classifierName, predictorNames, batch_size, data="processed"):
    ''' Run model on ensemble model
    '''
    # get test loader
    testLoader, inputSize = get_data_loader(mode="test", batch_size=batch_size, forClassifier=False, forEnsemble=True, data=data)

    # Loop through classifier models
    classifierCheckpoint = getBestModel(classifierName)
    modelClassifier = Binary_Classifier(inputSize)
    modelClassifier.load_state_dict(classifierCheckpoint)
    modelClassifier = modelClassifier.to("cuda")
    modelClassifier.eval()

    # loop through predictor models
    predictorModels = []
    for index, predModelName in enumerate(predictorNames):
        if not predModelName:
            continue
        predictorCheckpoint = getBestModel(predModelName)
        NETS = [Net, Net2, Net3, Net4]
        model = NETS[index](inputSize)
        model.load_state_dict(predictorCheckpoint)
        model = model.to("cuda")
        model.eval()
        predictorModels.append(model)


    # # load in the sub models used in the ensemble model
    # classifierCheckpoint = getBestModel("BinaryClassifier")
    # modelClassifier = Binary_Classifier(inputSize)
    # modelClassifier.load_state_dict(classifierCheckpoint)
    # modelClassifier = modelClassifier.to("cuda")
    # modelClassifier.eval()
    #
    # predictorCheckpoint1 = getBestModel("PredictorNet1")
    # modelPredictor1 = Net(inputSize)
    # modelPredictor1.load_state_dict(predictorCheckpoint1)
    # modelPredictor1 = modelPredictor1.to("cuda")
    # modelPredictor1.eval()
    #
    # predictorCheckpoint2 = getBestModel("PredictorNet2")
    # modelPredictor2 = Net2(inputSize)
    # modelPredictor2.load_state_dict(predictorCheckpoint2)
    # modelPredictor2 = modelPredictor2.to("cuda")
    # modelPredictor2.eval()
    #
    # predictorCheckpoint3 = getBestModel("PredictorNet3")
    # modelPredictor3 = Net3(inputSize)
    # modelPredictor3.load_state_dict(predictorCheckpoint3)
    # modelPredictor3 = modelPredictor3.to("cuda")
    # modelPredictor3.eval()
    #
    # predictorCheckpoint4 = getBestModel("PredictorNet4")
    # modelPredictor4 = Net4(inputSize)
    # modelPredictor4.load_state_dict(predictorCheckpoint4)
    # modelPredictor4 = modelPredictor4.to("cuda")
    # modelPredictor4.eval()

    # declare ensemble model
    model = Ensemble(modelClassifier, predictorModels)

    # move model to gpu
    model = model.to("cuda")

    criterion = MSLELoss()

    # set the model to eval
    model.eval()
    testLoss = 0

    # calc MAE
    MAE = nn.L1Loss()
    testMAE = 0

    accuracy = 0
    targets = []
    preds = []
    with tqdm(testLoader, desc="Testing", leave=False) as testBar:
        for input_x, target in testBar:
            # Move to gpu
            input_x = input_x.to("cuda")
            target = target.to("cuda")
            output = model(input_x.float())
            # convert output to int
            output = torch.round(output)
            # calculate loss
            loss = criterion(output, target)
            testLoss += loss.cpu().item() * len(target)
            testBar.set_postfix(loss = testLoss)
            # calculate MAE
            MAELoss = MAE(output, target)
            testMAE += MAELoss.cpu().item() * len(target)

            accuracy += (output.cpu().detach().numpy().round().astype(np.int) == target.cpu().numpy().round().astype(np.int)).mean()

            preds.extend(output.cpu().detach().numpy().round().astype(np.int).flatten().tolist())
            targets.extend(target.cpu().numpy().astype(np.int).flatten().tolist())

    # average the test loss
    testLoss /=  len(testLoader.dataset)
    accuracy /= int(len(testLoader.dataset)/batch_size)
    testMAE /= len(testLoader.dataset)

    print ("Accuracy: {}".format(accuracy))

    print ("Preds: {}".format(preds[:10]))
    print ("Targets: {}".format(targets[:10]))

    return testLoss, testMAE, preds, targets

def evaluate(classifierName, predictorName, batch_size, data="processed"):
    '''Evaluate the ensemble model with test data'''
    testLoss, testMAE, preds, targets = runTestOnEnsemble(classifierName, predictorName, batch_size, data=data)
    # save to file
    savePredToFile(preds, targets, "")
    # Scores
    print ("Test Loss: {}. Test MAE: {}".format(testLoss, testMAE))

def savePredToFile(preds, targets, modelName):
    modelFolder = "Models"
    savePath = os.path.join(modelFolder, modelName, "Ensemble_Preds.csv")
    with open(savePath, "w") as f:
        f.write("Pred, Target \n")
        for index, pred in tqdm(enumerate(preds), total = len(preds), desc="Saving Preds to {}".format(savePath)):
            f.write("{},{}\n".format(pred, targets[index]))

if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    evaluate(args.classifier, args.predictor, args.batch, args.data)
