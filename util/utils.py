import numpy as np
import torch.optim as optim
import tqdm
from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5, ChrNet6
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import gc 
import seaborn as sns
import torch
import datetime
from torch import nn

# Set seed for initializing network
## Not guaranteed to be identical across GPU-architectures!
## https://github.com/pytorch/pytorch/issues/84234
def seedEverything(seed=42):
    """
        Seeds everything for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# clear the cache and gpu memory
def clearCache():
    torch.cuda.empty_cache()
    gc.collect()


def loadModel(modelNumber, name="", input_size=500):
    """
    Loads the model based on the model number   
    """
    modelNumber = int(modelNumber)
    
    model_classes = {
        1: ChrNet1.Chromatin_Network1,
        2: ChrNet2.Chromatin_Network2,
        3: ChrNet3.Chromatin_Network3,
        4: ChrNet4.Chromatin_Network4,
        5: ChrNet5.Chromatin_Network5,
        6: ChrNet6.Chromatin_Network6
    }

    if modelNumber in model_classes:
        return model_classes[modelNumber](name, input_size)
    else:
        raise Exception("Invalid model number {}".format(modelNumber))


def trainModel(model, train_loader, test_loader, valid_loader, epochs, outputPath):
    """
        Runs the model, plots the training loss and accuracy, then tests the model and saves the model. this function returns the model, the loss values, and the accuracy values.

        Args:
            model: The model to run
            train_loader: The training data
            test_loader: The testing data
            valid_loader: The validation data
            epochs: The number of epochs to run
            criterion: The loss function
            optimizer: The optimizer
        
        returns:
            model: The model
            loss_values: The loss values
            accuracy_values: The accuracy values
        
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Name: {model.name}\t Device: {device}")
    model = model.to(device)

    best_accuracy = 0
    training_accuaracy = []
    training_loss = []
    testing_accuaracy = []
    test_auROC = []
    test_auPRC = []

    for epoch in tqdm.tqdm(range(epochs), leave=False, desc="Epoch", total=epochs):
        model.train()
        model, train_accuracy, train_loss = runEpoch(model, train_loader, criterion, optimizer)
        training_accuaracy.append(train_accuracy)
        training_loss.append(train_loss)

        model.eval()
        accuracy, auROC, auPRC, test_acc, test_loss = testModel(model, test_loader, criterion, device)
        testing_accuaracy.append(accuracy)
        test_auROC.append(auROC)
        test_auPRC.append(auPRC)

        plotAccuracy(training_accuaracy, "train_"+model.name, outputPath)
        plotAccuracy(testing_accuaracy, "test_"+model.name, outputPath)

        saveAccuracyCSV(training_accuaracy, outputPath+"/accuracy/train_"+model.name+".csv")
        saveAccuracyCSV(testing_accuaracy, outputPath+"/accuracy/test_"+model.name+".csv")

        saveAUCSV(test_auROC, outputPath+"/auROC_PRC/auROC_"+model.name+".csv")
        saveAUCSV(test_auPRC, outputPath+"/auROC_PRC/auPRC_"+model.name+".csv")

        saveLossPlot(training_loss, outputPath+"/loss/" + model.name + ".png")
        saveLossCSV(training_loss, outputPath+"/loss/" + model.name + ".csv")

    accuracy, auROC, auPRC = testModel(model, valid_loader, criterion, device, save=True)
    test_auPRC.append(auPRC)
    test_auROC.append(auROC)

    saveModelWeights(model, outputPath, model.name, epochs)
    saveAUCurve(test_auROC, test_auPRC, outputPath+"/auROC_PRC/" + model.name + ".png")

    return model.to("cpu")

# run the model for one epoch
def runEpoch(model, train_loader, criterion, optimizer):
    """
        Runs the model for one epoch

        Args:
            model: The model to run
            train_loader: The training data
            criterion: The loss function
            optimizer: The optimizer
        
        returns:
            model: The model
            accuracy: The accuracy
    """
    loss = 0
    accuracy = 0
    y_score = []
    y_true = []

    device = next(model.parameters()).device

    for inputs, labels in tqdm.tqdm(train_loader, desc="processing training batches", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item() * inputs.size(0)
        y_score.append(outputs.detach().cpu().numpy())
        y_true.append(labels.detach().cpu().numpy())

    loss /= len(train_loader.dataset)
    accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round())

    return model, accuracy, loss

# test the model
def testModel(model, test_loader, criterion, device, save=False):
    """
        Tests the model

        Args:
            model: The model to test
            test_loader: The testing data
            criterion: The loss function
        
        returns:
            accuracy: The accuracy
    """
    loss = 0
    accuracy = 0
    y_score = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="processing testing batches", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            loss += batch_loss.item() * inputs.size(0)
            y_score.append(outputs.detach().cpu().numpy())
            y_true.append(labels.detach().cpu().numpy())

    loss /= len(test_loader.dataset)
    accuracy = accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round())
    auROC = roc_auc_score(np.concatenate(y_true), np.concatenate(y_score))
    auPRC = average_precision_score(np.concatenate(y_true), np.concatenate(y_score))

    return accuracy, auROC, auPRC, loss

def saveAccuracyCSV(accuracy_data, path):
    df = pd.DataFrame(accuracy_data)
    df.to_csv(path)

def saveAUCSV(auc_data, path):
    df = pd.DataFrame(auc_data)
    df.to_csv(path)

def saveLossPlot(loss_data, path):
    import matplotlib.pyplot as plt
    plt.plot(loss_data)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.close()

def saveLossCSV(loss_data, path):
    df = pd.DataFrame(loss_data)
    df.to_csv(path)

def saveModelWeights(model, outputPath, name, epoch):
    os.makedirs(outputPath+"/modelWeights", exist_ok=True)
    torch.save(model.state_dict(), f"{outputPath}/modelWeights/{name}_epoch_{epoch}.pt")

def saveAUCurve(auROC_data, auPRC_data, path):
    import matplotlib.pyplot as plt
    plt.plot(auROC_data)
    plt.plot(auPRC_data)
    plt.legend(["auROC", "auPRC"])
    plt.title("auROC and auPRC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(path)
    plt.close()