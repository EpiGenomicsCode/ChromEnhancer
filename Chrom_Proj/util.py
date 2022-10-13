import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import OrderedDict

import gc
from sklearn import preprocessing
from sklearn import metrics as m

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

def train(trainer, batch_size, device, optimizer, model, loss_fn):
    """
        Trains the model with respect to the data
    """
    for train_loader in trainer:
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
            # Calculate the gradient
            loss.backward()
            # Update Weight
            optimizer.step()

def test(tester, batch_size, device, model):
    """
        Tests the model with respect to the data
    """
    for test_loader in tester:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)

        for test_data, test_label in tqdm(test_loader, desc="testing"):
            test_data, test_label = test_data.to(device), test_label.to(device)

            target = model(test_data)
            target = torch.flatten(target)
            test_label = torch.flatten(test_label)

def validate(model, validator, device):
    """
        Validates the model with respect to the data
    """
    allOut = []
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
    ##
    # PRINT fpr, tpr, pre, rec to file
    ##

    roc_auc = m.auc(fpr, tpr)
    data = list(OrderedDict.fromkeys(zip(pre,rec)))
    
    prc_auc = m.auc(sorted(pre), rec)

    plt.clf()
    plt.plot(pre, rec, color="darkgreen", 
            label='PRC curve (area = %0.2f' % prc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC Curve')
    plt.legend(loc="lower right")
    plt.savefig("output/prc/{}_prc.png".format(model.name))
    plt.clf()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("output/roc/{}_roc.png".format(model.name))
    allOut.append(target)
    
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

    return allOut

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

    # Train the model
    for epoch in range(epochs):     
        print("-----{}------".format(epoch)) 

        # Set model to train mode and train on training data
        model.train()
        train(trainer, batch_size, device, optimizer, model, loss_fn)
        model.eval()
        test(tester, batch_size, device, model)
        torch.save(model.state_dict(), savePath)
        gc.collect()
    
    print("Validating")
    validate(model, validator, device)


    print("\t\t\t{}".format((torch.cuda.memory_summary())))
    model = model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\t\t\t{}".format((torch.cuda.memory_summary())))