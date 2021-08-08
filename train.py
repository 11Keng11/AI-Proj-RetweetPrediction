'''
#Retweets Model trainer



'''
import torch
from Dataloader import get_data_loader
# from model import PCRNN # do we have model file?
from utils import *
import os
import time
import math
from tqdm import tqdm
import argparse
from Regression_NN_1 import *
from glob import glob
import re

## PYTORCH DEBUG TOOLS
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

#========== ARGPARSE BLOCK ==========#
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", help="num of epochs for training", type=int, default=100)
parser.add_argument("-b", "--batch", help="batch size for training", type=int, default=4096)
parser.add_argument("-n", "--name", help="training session name", type=str, default="training1")
parser.add_argument("-lr", "--learningrate", help="learning rate for model training", type=float, default=1e-3)
parser.add_argument("-o", "--optimizer", help="optimizer for model training", type=str, choices=["SGD", "ADAM", "RMSPROP"], default="SGD")
parser.add_argument("-c", "--classifier", type=str_to_bool, default=False)
parser.add_argument("-d", "--data", type=str, default="processed", help="Dataset prestring")

def parserSummary(args):
    print ("Training #Retweets Model.")
    print ("Using dataset pre string: {}".format(args.data))
    print ("Running with: {} epoch(s)".format(args.epoch))
    print ("Batch Size: {}".format(args.batch))
    print ("Learning Rate: {}".format(args.learningrate))
    print ("Optimizer: {}".format(args.optimizer))
    print ("For classifier: {}".format(args.classifier))
    print ("Training session saved as: {}".format(args.name))

#======= END OF ARGPARSE BLOCK =======#


#======= HELPER FUNCTIONS BLOCK =======#
def initTrainingSession(sessionName):
    ''' Function calls to make the model directory for model
    saving purposes with session name as input'''
    modelFolder = "Models"
    RESUME_TRAIN = False
    if not os.path.isdir(modelFolder):
        # folder for saving models dont exist yet
        os.mkdir(modelFolder)
        print ("{} folder created as it does not exist!".format(modelFolder))
    saveDir = os.path.join(modelFolder, sessionName)
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)
        print ("{} folder created as it does not exist!".format(saveDir))
    if checkIfTrainingSessionExists(sessionName):
        check = input("Continue? (Y/N) > ")
        if check == "N" or check == "n":
            quit()
        elif check == "Y" or check == "y":
            print ("Resume Training: TRUE")
            RESUME_TRAIN = True

    print ("\nTraining session successfully inited!")
    return RESUME_TRAIN

def checkIfTrainingSessionExists(modelName):
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName)
    savePath = os.path.join(saveDir, "trainingStats.csv")
    if os.path.exists(savePath):
        print ("EXISTING TRAINING SESSION EXISTS!")
        return True
    return False

def saveRunArgs(args):
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, args.name)
    savePath = os.path.join(saveDir, "arguments.txt")
    with open(savePath, "w") as f:
        for key in vars(args):
            f.write("{}: {}\n".format(key, getattr(args, key)))

def saveTrainingStats(epoch, trainLoss, valLoss, modelName):
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName)
    savePath = os.path.join(saveDir, "trainingStats.csv")
    # make csv file for saving model training stats if it does not exist
    if not os.path.exists(savePath):
        with open(savePath, "a", encoding="utf-8") as f:
            headers = ["epoch", "train loss", "val loss"]
            f.write(",".join(headers))
            f.write("\n")

    # finally open the csv file and add stuff to it
    with open(savePath, "a", encoding="utf-8") as f:
        data = ",".join([str(epoch), str(trainLoss), str(valLoss)])
        f.write(data+"\n")

def saveModel(model, optimizer, scheduler, trainLoss, valLoss, modelName, epoch):
    '''Saves the current model state dict based on modelname input
     and current epoch'''
    modelFolder = "Models"
    modelSavesFolder = "ModelSaves"
    saveDir = os.path.join(modelFolder, modelName, modelSavesFolder)
    # save the model
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    savePath = os.path.join(saveDir, "{}_epoch_{}.pth".format(modelName, epoch) )
    torch.save({
                "Epoch": epoch,
                "Model": model.state_dict(),
                "Optimizer": optimizer.state_dict(),
                "Scheduler": scheduler.state_dict(),
                "trainLoss": trainLoss,
                "valLoss": valLoss,
                }, savePath)

def saveBestModel(model, modelName):
    '''Specific function to save the best model so far'''
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName, "BestModel.pth")
    torch.save(model.state_dict(), saveDir)

def getChosenOptimizer(opt):
    '''Get corresponding pytorch optimizer based on argparse input'''
    if opt == "SGD":
        return torch.optim.SGD
    elif opt == "ADAM":
        return torch.optim.Adam
    elif opt == "RMSPROP":
        return torch.optim.RMSprop
    else:
        raise Exception("Invalid optimizer input!")
#==== END OF HELPER FUNCTIONS BLOCK ====#

def train(RESUME_TRAIN, n_epochs=1, batch_size=64, lr=1e-3, o=torch.optim.SGD, experiment_name="test", forClassifier=False, data="processed"):
    '''Train'''

    # get the respective data loaders
    trainLoader, inputSize = get_data_loader(mode="train", batch_size=batch_size, forClassifier=forClassifier, forEnsemble=False, data=data)
    valLoader, _ = get_data_loader(mode="val", batch_size=batch_size, forClassifier=forClassifier, forEnsemble=False, data=data)

    # make model
    if forClassifier:
        model = Binary_Classifier(inputSize)
    else:
        model = Net(inputSize)

    # move model to gpu
    model = model.to("cuda")

    # declare optimizer
    optimizer = o(model.parameters(), lr=lr)

    # set up the lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr,step_size_up=5,mode="triangular2", gamma=0.1, cycle_momentum=False)

    # declare init vars
    start_epoch = 1
    bestValLoss = 1e20

    if RESUME_TRAIN:
        # we have to load in the model.pth
        checkpoint = getModelCheckpoint(experiment_name)
        model.load_state_dict(checkpoint["Model"])
        optimizer.load_state_dict(checkpoint["Optimizer"])
        scheduler.load_state_dict(checkpoint['Scheduler'])
        start_epoch = checkpoint["Epoch"] + 1
        bestValLoss = checkpoint["valLoss"]

    if forClassifier:
        criterion = nn.BCELoss()
    else:
        criterion = MSLELoss()


    with tqdm(range(start_epoch, start_epoch + n_epochs)) as ebar:

        for epoch in ebar:
            # training section
            trainLoss = 0
            model.train()
            with tqdm(trainLoader, desc="Epoch: {} Train".format(epoch), leave=False) as tbar:
                for input_x, target in tbar:
                    optimizer.zero_grad()
                    # Move to gpu
                    input_x = input_x.to("cuda")
                    target = target.to("cuda")
                    # calculate prediction
                    pred = model(input_x.float())
                    loss = criterion(pred, target.float())

                    loss.backward()

                    optimizer.step()


                    trainLoss += loss.cpu().item()
                    tbar.set_postfix(loss = trainLoss)

            scheduler.step()

            # Evaluation
            # accuracy = 0
            valLoss = 0
            model.eval()
            with tqdm(valLoader, desc="Epoch: {} Valid".format(epoch), leave=False) as vbar:
                for input_x, target in vbar:
                    # Move to GPU
                    input_x = input_x.to("cuda")
                    target = target.to("cuda")
                    pred = model(input_x.float())
                    loss = criterion(pred, target.float())
                    valLoss += loss.cpu().item()
                    pred = pred.cpu()
                    # accuracy += torch.sum(pred.cpu().int() == target.cpu().int()).item()
                    # print ("PRED: ", pred)
                    # for index, p in enumerate(pred):
                    #     if int(p) == target[index]:
                    #         accuracy += 1
                    vbar.set_postfix(loss = valLoss)

            # print (accuracy)

            trainLoss /= int(len(trainLoader.dataset)/batch_size)
            valLoss /= int(len(valLoader.dataset)/batch_size)
            # accuracy /= len(valLoader.dataset)

            ebar.set_description("Epoch: {:3}/{:3} Train Loss: {:.4f} Validation Loss: {:.4f}".format(
                epoch, start_epoch+n_epochs-1, trainLoss, valLoss))

            if valLoss < bestValLoss:
                bestValLoss = valLoss
                saveBestModel(model, experiment_name)

            # save model every epoch for resuming in future
            saveModel(model, optimizer, scheduler, trainLoss, valLoss, experiment_name, epoch)

            # save states
            saveTrainingStats(epoch, trainLoss, valLoss, experiment_name)




if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    # get Optimizer
    optimizer = getChosenOptimizer(args.optimizer)
    # initialize the training session
    RESUME_TRAIN = initTrainingSession(args.name)
    # save running arguments
    saveRunArgs(args)
    # train!
    try:
        train(RESUME_TRAIN, args.epoch, args.batch, args.learningrate, optimizer, args.name, args.classifier, args.data)
    except KeyboardInterrupt:
        exit()
