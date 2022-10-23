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

import pdb

def readfiles(id, chromType, label, file_location):
    """
        Reads preprocessed files and returns the data and labels

        input:
        =====
            id: string
            chromType: list of strings
            label: string: Enhancer labels
            file_location: string
    """
    print(file_location)
    files = glob(file_location)
    labels = []
    data = {}

    for fileType in chromType:
        filename = [
            i for i in files if id in i and fileType in i and "chromtrack" in i and label in i]
        assert len(filename) != 0
        print("Processing: {}".format(filename[0]))
        fileName = filename[0]

        data[fileType] = pd.read_csv(fileName, delimiter=" ",  header=None)

    horizontalConcat = pd.DataFrame()
    for fileType in chromType:
        horizontalConcat = pd.concat(
            [horizontalConcat, data[fileType]], axis=1)

    labelFileName = [
                        i for i in files if ".label" in i and id in i and label in i
                    ]
    assert len(labelFileName) > 0
    print("Processing: {}".format(labelFileName[0]))
    label = pd.read_csv(labelFileName[0], delimiter=" ", header=None)

    return np.array(horizontalConcat.values[:]), np.array(label.values[:])

def getData(chromtypes, 
            id, 
            trainLabel, 
            testLabel, 
            validLabel,
            fileLocation="./Data/220802_DATA"
        ):
    """
    Returns the training, testing and validation data based on the input

    Input:
        chromtypes: List of String that represent the order of the chromatine types 
            (ex: ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"])
        
        id: String contaning the whole Chromatine Cell identification 
            (ex: "A549")
        
        trainLabel: String containing the training Label 
            (ex: "chr10-chr17")
        
        testLabel: String containing the test Label 
            (ex: "chr10")
        
        validLabel: String contatining the validation labels 
            (ex: "chr11")
        
        fileLocation: Relative file path for where the files are being saved 
            (ex: ./Data/220708/DATA)

    Return:
        trainer: list of the training data
        
        tester: list of the testing data
        
        validator: list of the validation data
    """
    
    chr_train = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=trainLabel,
        file_location=fileLocation+"/TRAIN/*")

    chr_test = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=testLabel,
        file_location=fileLocation+"/HOLDOUT/*")

    chr_valid = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=validLabel,
        file_location=fileLocation+"/HOLDOUT/*")

    trainer = [chr_train]
    tester = [chr_test]
    validator = [chr_valid]

    return trainer, tester, validator   


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
        testloss = 0
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

    plotROC(model, fpr, tpr)
    plotPRC(model, pre, rec)
    writeData(model, pre, rec, fpr, tpr)

    return valid_loader.labels, target

def writeData(model, pre, rec, fpr, tpr):
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
    f = open("output/coord/pre_{}.csv".format(model.name), "w+")
    f.write("pre\n")
    data = ""
    for i in pre:
        data += str(i) + ","
    f.write(data+"\n")
    f.close()

    f = open("output/coord/rec_{}.csv".format(model.name), "w+")
    f.write("rec\n")
    data = ""
    for i in rec:
        data += str(i) + ","
    f.write(data+"\n")
    f.close()
    
    
    f = open("output/coord/fpr_{}.csv".format(model.name), "w+")
    f.write("fpr\n")
    data = ""
    for i in fpr:
        data += str(i) + ","
    f.write(data+"\n")
    f.close()


    f = open("output/coord/tpr_{}.csv".format(model.name), "w+")
    f.write("tpr\n")
    data = ""
    for i in tpr:
        data += str(i) + ","
    f.write(data+"\n")
    f.close()

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
    savePath = "output/model_weight_bias/model_{}.pt".format(model.name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)
    Loss = []
    # Train the model
    for epoch in range(epochs):     
        print("-----{}------".format(epoch)) 
        # Set model to train mode and train on training data
        model.train()
        trainLoss = train(trainer, batch_size, device, optimizer, model, loss_fn)
        model.eval()
        testLoss = test(tester, batch_size, device, model, loss_fn)
        Loss.append((trainLoss, testLoss))
        torch.save(model.state_dict(), savePath)
        gc.collect()
    
    f = open("./output/rmseTest/loss_Per_epoch_{}.txt".format(model.name), "w+")
    f.write(str(Loss))
    f.close()



    print("Validating")
    realValid, predictedValid = validate(model, validator, device)


    # print("\t\t\t{}".format((torch.cuda.memory_summary())))
    model = model.to("cpu")
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    # print("\t\t\t{}".format((torch.cuda.memory_summary())))

    return realValid, predictedValid,  model