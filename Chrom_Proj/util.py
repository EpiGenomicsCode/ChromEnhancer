import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader



def input_model(data, batch_size, optimizer, model, loss_fn, work="train"):
    """
        Trains the model with respect to the data
    """
    totalLoss = []
    allLabels = []
    alltargets = []
    labels = []
    targets = []
    for loader in data:        
        if work == "valid":
            loader.drop = None

        loader.loadChunk()        
        loader = DataLoader(loader, shuffle=True, batch_size=batch_size)
        loaderLoss = 0

        for data, label in tqdm(loader):
            # Clear gradients
            optimizer.zero_grad()

            # Forward Pass
            target = model(data)

            # Calculate the Loss
            loss = loss_fn(
                        target.to(torch.float32), 
                        label.to(torch.float32)
                        )
            if work == "train":
                # update loss
                loss.backward()
                # Update Weight
                optimizer.step()
            
            # save the loss
            loaderLoss += loss.item() * data.size(0)
        
            # Clean the data
            target = torch.flatten(target)
            label = torch.flatten(label)
        
            labels.append(label.cpu())
            targets.append(target.cpu())
            

        totalLoss.append(loaderLoss)
        
    if work == "validate":
        labels = torch.cat(labels).detach().numpy()
        targets = torch.cat(targets).detach().numpy()

        fpr, tpr, _ =  m.roc_curve(labels, targets)
        pre, rec, _ = m.precision_recall_curve(labels, targets)

        ROCAUC = plotROC(model, fpr, tpr)
        PRCAUC = plotPRC(model, pre, rec)
        writeData(model, pre, rec, fpr, tpr, ROCAUC, PRCAUC)
    
    totalLoss = np.sum(totalLoss)/len(loader)
    print("\t{} Loss: {}".format(work, totalLoss) )
    clearTorch()
    return totalLoss


def clearTorch():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.device.type  =="cuda":
                    obj = obj.cpu()
        except:
            pass
    
    torch.cuda.empty_cache()
    gc.collect()

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
    elif modelType == 2:
        model = Chromatin_Network2(name)   
    elif modelType == 3:
        model = Chromatin_Network3(name)   
    elif modelType == 4:
        model = Chromatin_Network4(name)   
    else:
        print("model does not exist")
        quit()

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
    elif modelType == 2:
        model = Chromatin_Network2("validator")  
    elif modelType == 3:
        model = Chromatin_Network3("validator")  
    elif modelType == 4:
        model = Chromatin_Network4("validator")  
    else:
        print("no model found")
        quit()

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
    trainLoss = []
    testLoss = []
    # Train the model
    for epoch in range(epochs):     
        print("-----Epoch: {}------".format(epoch)) 
        clearTorch()
        model = model.to(device)
        # Set model to train mode and train on training data
        model.train()
        trainLossEpoch = input_model(trainer, batch_size, optimizer, model, loss_fn, work="train")
        model.eval()
        testLossEpoch = input_model(tester, batch_size, optimizer, model, loss_fn, work="test")
        trainLoss.append(trainLossEpoch)
        testLoss.append(testLossEpoch)
    
           
    os.makedirs("./output/Info/", exist_ok=True)
    f = open("./output/Info/Loss_{}.txt".format(model.name), "w+")
    f.write("trainLoss: {}\n".format(str([i for i in trainLoss]).replace("\n","").replace(" ", "")))
    f.write("testLoss: {}".format(str([i for i in testLoss]).replace("\n","").replace(" ", "")))
    f.close()

    print("Validating")
    validLoss = input_model(validator, batch_size, optimizer, model, loss_fn, work="validate")
    model = model.to("cpu")
    torch.save(model.state_dict(), savePath) 

    