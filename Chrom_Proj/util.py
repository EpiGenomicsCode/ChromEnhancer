import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Chrom_Proj.visualizer import *
import Chrom_Proj as CP
import gc
from sklearn import preprocessing
from sklearn import metrics as m
import sklearn.metrics as sm
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
from Chrom_Proj.model import *
import pdb
import os


def train(trainer, batch_size, device, optimizer, model, loss_fn):
    """
        Trains the model with respect to the data
    """
    totalLoss = []
    for train_loader in trainer:
        trainLoss = 0
        if train_loader != CP.chrom_dataset.Chromatin_Dataset:
            train_loader = DataLoader(
                train_loader, batch_size=batch_size, shuffle=True)

        for data, label in tqdm(train_loader, desc="training"):
            # Load data appropriatly
            data, label = data.to(device), label.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Clean the data
            target = torch.flatten(target)
            label = torch.flatten(label)
            # Calculate the Lo
            loss = loss_fn(
                target.to(
                    torch.float32), label.to(
                    torch.float32))
            # save the loss
            trainLoss += loss.item() * data.size(0)

            # Calculate the gradient
            loss.backward()
            # Update Weight
            optimizer.step()
        totalLoss.append(trainLoss)
    totalLoss = np.sum(totalLoss)/len(trainer)
    print("\tTrain Loss: {}".format(totalLoss) )
    return totalLoss

def test(tester, batch_size, device, model, loss_fn):
    """
        Tests the model with respect to the data
    """
    
    totalLoss = []
    for test_loader in tester:
        testLoss = 0
        if test_loader != CP.chrom_dataset.Chromatin_Dataset:
            test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)

        for test_data, test_label in tqdm(test_loader, desc="testing"):
            test_data, test_label = test_data.to(device), test_label.to(device)
            target = model(test_data)
            target = torch.flatten(target)
            label = torch.flatten(test_label)
            # Calculate the Lo
            loss = loss_fn(
                target.to(
                    torch.float32), label.to(
                    torch.float32))
            testLoss += loss.item() * test_data.size(0)
        totalLoss.append(testLoss)
        
    totalLoss = np.sum(totalLoss)/len(tester)

    print("\tTest Loss: {}".format(totalLoss) )
    return totalLoss
                                    
def validate(model, validator, device):
    """
        Validates the model with respect to the data

        INPUT:
        =====
            model: pytorch model: for validation
            validator: data to validate
            device: the device to run on
        RETURNS:
            targetData: list of realdata
            target: list of predicted data
    """
    predictedData = []
    model = model.to("cpu")

    for valid_loader in validator:
        # Set model to validation
        model.eval()
        target = []
        for data in tqdm(valid_loader.data, desc="validating"):
            target.append(model(torch.tensor(np.array([data]), dtype=torch.float32).to("cpu")))

    target = torch.tensor(target)
    target = torch.flatten(target)
    fpr, tpr, _ = m.roc_curve(valid_loader.labels, target)
    pre, rec, _ = m.precision_recall_curve(valid_loader.labels, target)

    predictedData.append(target)    

    ROCAUC = plotROC(model, fpr, tpr)
    PRCAUC = plotPRC(model, pre, rec)
    writeData(model, pre, rec, fpr, tpr, ROCAUC, PRCAUC)

    return valid_loader.labels, target

def writeData(model, pre, rec, fpr, tpr, ROCAUC, PRCAUC):
    """
        Writes the PCR and ROC curve data to file

        Input:
        ======
            model: Pytorch model

            pre: array: precision data
            
            rec: array: recall data

            fpr: array: false positive rate

            tpr: array: true positive rate
    """
    f = open("output/Info/Analysis_{}.csv".format(model.name), "w+")
    f.write("\npre:" + str([i for i in pre]).replace(" ", "").replace("\n", ""))
    f.write("\nrec:" + str([i for i in rec]).replace(" ", "").replace("\n", ""))
    f.write("\nfpr:" + str([i for i in fpr]).replace(" ", "").replace("\n", ""))
    f.write("\ntpr:"+str([i for i in tpr]).replace(" ", "").replace("\n", ""))
    f.write("\nROCAUC:"+str(ROCAUC))
    f.write("\nPRCAUC:"+str(PRCAUC))

    f.close()

def readAnalysisData(fileName):
    """
        reads in the pre, rec, fpr and tpr data from the file given
    """
    print("processing: {}".format(fileName))
    file = open(fileName, 'r')

    data = {"pre":[],"rec":[],"fpr":[],"tpr":[],"PRCAUC":0, "ROCAUC":0, "trainLoss":[], "testLoss":[]}
    
    for line in file:
        line = line.strip()
        if len(line) > 1:
            if "pre" in line:
                data["pre"] = eval(line[line.index(":")+1:].strip())
            if "fpr" in line:
                data["fpr"] = eval(line[line.index(":")+1:].strip())
            if "rec" in line:
                data["rec"] = eval(line[line.index(":")+1:].strip())
            if "tpr" in line:
                data["tpr"] = eval(line[line.index(":")+1:].strip())
            if "PRCAUC" in line:
                data["PRCAUC"] = float(line[line.index(":")+1:].strip())
            if "ROCAUC" in line:
                data["ROCAUC"] = float(line[line.index(":")+1:].strip())
    return data

def readLossData(fileName):
    print("processing: {}".format(fileName))
    file = open(fileName, 'r')
    data = {"trainLoss": [], "testLoss":[]}
    for line in file:
        if "trainLoss" in line:
            data["trainLoss"] = eval(line[line.index(":")+1:].strip())
        
        if "testLoss" in line:
            data["testLoss"] = eval(line[line.index(":")+1:].strip())
        
    return data

def loadModel(modelType, name):
    model = None
    if modelType == 1:
        model = Chromatin_Network1(name)
    if modelType == 2:
        model = Chromatin_Network2(name)   
    if modelType == 3:
        model = Chromatin_Network3(name)   
    if modelType == 4:
        model = Chromatin_Network4(name)   
    if modelType == 5:
        model = Chromatin_Network5(name)   
    if modelType == 6:
        model = Chromatin_Network6(name)  

    return model

def loadModelfromFile(modelFileName, modelType):
    """
    Loads the models architecture from a specific file location

    Inputs:
        modelFileName: filename of the model
        modelType: the architechture of the model
    
    Returns:
        model: Pytorch Model with architecture
    """
    if modelType == 1:
        model = Chromatin_Network1("validator")
    if modelType == 2:
        model = Chromatin_Network2("validator")  
    if modelType == 3:
        model = Chromatin_Network3("validator")  
    if modelType == 4:
        model = Chromatin_Network4("validator")  
    if modelType == 5:
        model = Chromatin_Network5("validator")  
    if modelType == 6:
        model = Chromatin_Network6("validator")  

    model.load_state_dict(torch.load(modelFileName,map_location=torch.device('cpu')))
    
    return model

def runModel(
        trainer,
        tester,
        validator,
        model,
        optimizer,
        loss_fn,
        batch_size,
        epochs):
    """
        Runs the model whole with respect to the data
    """
    os.makedirs("./output/model_weight_bias/",exist_ok=True)
    savePath = "output/model_weight_bias/model_{}.pt".format(model.name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)
    trainLoss = []
    testLoss = []
    # Train the model
    for epoch in range(epochs):     
        print("-----{}------".format(epoch)) 
        # Set model to train mode and train on training data
        model.train()
        trainLossEpoch = train(trainer, batch_size, device, optimizer, model, loss_fn)
        model.eval()
        testLossEpoch = test(tester, batch_size, device, model, loss_fn)
        trainLoss.append(trainLossEpoch)
        testLoss.append(testLossEpoch)
        torch.save(model.state_dict(), savePath)
        gc.collect()
    
    os.makedirs("./output/Info/", exist_ok=True)
    f = open("./output/Info/Loss_{}.txt".format(model.name), "w+")
    f.write("trainLoss: {}\n".format(str([i for i in trainLoss]).replace("\n","").replace(" ", "")))
    f.write("testLoss: {}".format(str([i for i in testLoss]).replace("\n","").replace(" ", "")))
    f.close()

    print("Validating")
    realValid, predictedValid = validate(model, validator, device)

    model = model.to("cpu")

    return realValid, predictedValid,  model