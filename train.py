'''
#Retweets Model trainer



'''
import torch
from Dataloader import get_data_loader
# from model import PCRNN # do we have model file?
from utils import RMSLELoss, MSLELoss # custom loss function
import os
import time
import math
from tqdm import tqdm
import argparse
from Regression_NN_1 import Net

## PYTORCH DEBUG TOOLS
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

#========== ARGPARSE BLOCK ==========#
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", help="num of epochs for training", type=int, default=100)
parser.add_argument("-b", "--batch", help="batch size for training", type=int, default=64)
parser.add_argument("-n", "--name", help="training session name", type=str, default="training1")
parser.add_argument("-lr", "--learningrate", help="learning rate for model training", type=float, default=1e-3)
parser.add_argument("-o", "--optimizer", help="optimizer for model training", type=str, choices=["SGD", "ADAM", "RMSPROP"], default="SGD")

def parserSummary(args):
    print ("Training #Retweets Model.")
    print ("Running with: {} epoch(s)".format(args.epoch))
    print ("Batch Size: {}".format(args.batch))
    print ("Learning Rate: {}".format(args.learningrate))
    print ("Optimizer: {}".format(args.optimizer))
    print ("Training session saved as: {}".format(args.name))

#======= END OF ARGPARSE BLOCK =======#

#======= HELPER FUNCTIONS BLOCK =======#
def initTrainingSession(sessionName):
    ''' Function calls to make the model directory for model
    saving purposes with session name as input'''
    modelFolder = "Models"
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
            modelFolder = "Models"
            saveDir = os.path.join(modelFolder, sessionName)
            savePath = os.path.join(saveDir, "trainingStats.csv")
            os.remove(savePath)
            print ("Removed: {}".format(savePath))

    print ("\nTraining session successfully inited!")

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


def saveTrainingStats(epoch, trainLoss, valLoss, accuracy, modelName):
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName)
    savePath = os.path.join(saveDir, "trainingStats.csv")
    # make csv file for saving model training stats if it does not exist
    if not os.path.exists(savePath):
        with open(savePath, "a", encoding="utf-8") as f:
            headers = ["epoch", "train loss", "val loss", "accuracy"]
            f.write(",".join(headers))
            f.write("\n")

    # finally open the csv file and add stuff to it
    with open(savePath, "a", encoding="utf-8") as f:
        data = ",".join([str(epoch), str(trainLoss), str(valLoss), str(accuracy)])
        f.write(data+"\n")

def saveModel(model, modelName, epoch):
    '''Saves the current model state dict based on modelname input
     and current epoch'''
    modelFolder = "Models"
    saveDir = os.path.join(modelFolder, modelName)
    # save the model
    savePath = os.path.join(saveDir, "{}_epoch_{}.pth".format(modelName, epoch) )
    torch.save(model.state_dict(), savePath)
    # print ("Model: {} saved in {}".format(modelName, savePath))

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

def train(model=None, n_epochs=1, batch_size=64, lr=1e-3, o=torch.optim.SGD, experiment_name="test"):
    '''Train the model given'''
    if model is None:
        raise Exception("MODEL not provided!")

    # get the respective data loaders
    trainLoader = get_data_loader(mode="train", batch_size=batch_size)
    valLoader = get_data_loader(mode="val", batch_size=batch_size)

    # declare optimizer and criterion
    optimizer = o(model.parameters(), lr=lr)
    # criterion = RMSLELoss()
    criterion = MSLELoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    bestAcc = 0
    # move model to gpu
    model = model.to("cuda")

    with tqdm(range(1, n_epochs+1)) as ebar:
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
                    # print("INPUT: ", input_x)
                    # print ("Input finite: ", torch.isfinite(input_x).all())
                    pred = model(input_x.float())
                    # print ("PRED: ", pred)
                    # print ("TARGET: ", target)
                    loss = criterion(pred, target.float())
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad:
                    #         print ("layer finite: ", torch.isfinite(param).all())
                    # # print ("Before backward!")
                    # print ("Loss: {}".format(loss.data))
                    loss.backward()
                    # print ("After backward!")
                    # print ("Loss: {}, grad: {}".format(loss.data, loss.grad))
                    optimizer.step()
                    # print ("After step")
                    # print ("Loss: {}, grad: {}".format(loss.data, loss.grad))
                    trainLoss += loss.cpu().item()
                    pred = pred.cpu()
                    # print ("PRED: ", pred)
                    tbar.set_postfix(loss = trainLoss)

            # Evaluation
            accuracy = 0
            validLoss = 0
            model.eval()
            with tqdm(valLoader, desc="Epoch: {} Valid".format(epoch), leave=False) as vbar:
                for input_x, target in vbar:
                    # Move to GPU
                    input_x = input_x.to("cuda")
                    target = target.to("cuda")
                    pred = model(input_x.float())
                    loss = criterion(pred, target.float())
                    validLoss += loss.cpu().item()
                    pred = pred.cpu()
                    accuracy += torch.sum(pred.cpu().int() == target.cpu().int()).item()
                    # print ("PRED: ", pred)
                    # for index, p in enumerate(pred):
                    #     if int(p) == target[index]:
                    #         accuracy += 1
                    vbar.set_postfix(loss = validLoss)

            # print (accuracy)

            trainLoss /= int(len(trainLoader.dataset)/batch_size)
            validLoss /= int(len(valLoader.dataset)/batch_size)
            accuracy /= len(valLoader.dataset)

            ebar.set_description("Epoch: {:3}/{:3} Train Loss: {:.4f} Validation Loss: {:.4f} Accuracy: {:.2f}%".format(
                epoch, n_epochs, trainLoss, validLoss, accuracy*100))

            # if accuracy > bestAcc:
            #     bestAcc = accuracy
            saveModel(model, experiment_name, epoch)

            # save states
            saveTrainingStats(epoch, trainLoss, validLoss, accuracy, experiment_name)




if __name__ == "__main__":
    args = parser.parse_args()
    parserSummary(args)
    # get Optimizer
    optimizer = getChosenOptimizer(args.optimizer)
    # initialize the training session
    initTrainingSession(args.name)
    # save running arguments
    saveRunArgs(args)
    # load in model
    # model = Net(10, 64)
    model = Net(58, 64)
    # train!
    try:
        train(model, args.epoch, args.batch, args.learningrate, optimizer, args.name)
    except KeyboardInterrupt:
        exit()
