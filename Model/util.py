from cProfile import label
import re
from Model.model import Chromatin_Network
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from sklearn import metrics as m



from sklearn import preprocessing

metrics = {}

metrics["trainLoss"] = []

metrics["trainF1"] = []
metrics["testF1"] = []

metrics["trainPrec"] = []
metrics["testPrec"] = []

metrics["trainAcc"] = []
metrics["testAcc"] = []


def readfiles(id, chromType, label, file_location):
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

    # print("Before remove Duplicate")
    # print(horizontalConcat.shape)
    # horizontalConcat = horizontalConcat.drop_duplicates(keep="last").round(5)
    # print(horizontalConcat.shape)
    # print("After remove Duplicate")

    return np.array(horizontalConcat.values), np.array(label.values)

def fitSVM(supportvectormachine, epochs, train, test, valid):
    print("Running the SVM")
    for t in train:
        supportvectormachine.fit(t.data, t.labels)

    for t in test:
        print("\ntest:{}".format(supportvectormachine.score(t.data, t.labels)))

    for t in valid:
        print(
            "\nvalidation:{}".format(
                supportvectormachine.score(
                    t.data,
                    t.labels)))

    return supportvectormachine

def plotPCA(chrom):
    print("Processing PCA for {}".format(chrom.filename))
    x = chrom.data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=[
            'principal component 1',
            'principal component 2'])
    labels = pd.DataFrame(chrom.labels, columns=["target"])
    finalDf = pd.concat([principalDf, labels], axis=1)
    finalDf.to_csv("./Data/PCA_compression.csv")

    colors = ['b', 'r']
    targets = [0, 1]
    ax1 = plt.subplot(211)

    ax1.set_ylabel('Principal Component 2', fontsize=15)
    ax1.set_title('2 component PCA, target = 0', fontsize=20)

    indicesToKeep = finalDf['target'] == 0
    ax1.scatter(finalDf.loc[indicesToKeep,
                            'principal component 1'],
                finalDf.loc[indicesToKeep,
                'principal component 2'],
                c='r',
                s=50)

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.set_title('2 component PCA, target = 1', fontsize=20)
    ax2.set_xlabel('Principal Component 1', fontsize=15)

    indicesToKeep = finalDf['target'] == 1
    ax2.scatter(finalDf.loc[indicesToKeep,
                            'principal component 1'],
                finalDf.loc[indicesToKeep,
                'principal component 2'],
                c='b',
                s=50)
    plt.tight_layout()
    plt.savefig("output/PCA_" + chrom.filename + ".png")

def binary_acc(y_pred, y_real):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_real).sum().float()
    acc = correct_results_sum/y_real.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def updateMetric(pred, real, method="train"):
    global metrics
    pred = pred.detach().cpu()
    real = real.detach().cpu()

    if method=="train":
        metrics["trainAcc"][-1] += binary_acc(pred, real)
        metrics["trainF1"][-1] += m.f1_score(real, pred)
        metrics["trainPrec"][-1] += m.precision_score(real, pred)
    else:
        metrics["testAcc"][-1] += binary_acc(pred, real)
        metrics["testF1"][-1] += m.f1_score(real, pred)
        metrics["testPrec"][-1] += m.precision_score(real, pred)

def plotMetric():
    global metrics
    for i in metrics.keys():
        print("{}\t\t{}".format(i, np.round(metrics[i],3)))
        plt.clf()
        plt.plot(metrics[i], label=i)
        plt.savefig("output/{}.png".format(i))


    plt.clf()
    plt.plot(metrics["trainAcc"], label="train")
    plt.plot(metrics["testAcc"], label="test")
    plt.legend()
    plt.savefig("output/Acc.png")
    
    plt.clf()
    plt.plot(metrics["trainF1"], label="train")
    plt.plot(metrics["testF1"], label="test")
    plt.legend()
    plt.savefig("output/F1.png")
    
    plt.clf()
    plt.plot(metrics["trainPrec"], label="train")
    plt.plot(metrics["testPrec"], label="test")
    plt.legend()
    plt.savefig("output/Prec.png")
    
    plt.clf()
    plt.plot(metrics["trainLoss"], label="loss")
    plt.legend()
    plt.savefig("output/loss.png")
    

def train(trainer, batch_size, device, optimizer, model, loss_fn):
    global metrics

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
            metrics["trainLoss"][-1]+=loss.item()
            # Calculate the gradient
            loss.backward()
            # Update Weight
            optimizer.step()
            updateMetric(target, label, "train")

def test(tester, batch_size, device, model):
    global metrics
    for test_loader in tester:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)

        for test_data, test_label in tqdm(test_loader, desc="testing"):
            test_data, test_label = test_data.to(device), test_label.to(device)

            target = model(test_data)
            target = torch.flatten(target)
            test_label = torch.flatten(test_label)
            updateMetric(target, test_label, "test")


def validate(model, validator, device):
    allOut = []
    for valid_loader in validator:
        # Set model to validation
        model.eval()
        target = model(torch.tensor(valid_loader.data, dtype=torch.float32).to(device))
        target = torch.flatten(target)
        fpr, tpr, _ = m.roc_curve(valid_loader.labels, target.detach().cpu())

        roc_auc = m.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("output/roc.png")
        allOut.append(target)
    return allOut

def loadModel(modelFileName="output/model.pt"):
    model = Chromatin_Network()
    model.load_state_dict(torch.load(modelFileName))
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
    global metrics
    savePath = "output/model.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)

    # Train the model
    for epoch in range(epochs):      
        metrics["trainLoss"].append(0)

        metrics["trainF1"].append(0)
        metrics["testF1"].append(0)

        metrics["trainPrec"].append(0)
        metrics["testPrec"].append(0)

        metrics["trainAcc"].append(0)
        metrics["testAcc"].append(0)


        # Set model to train mode and train on training data
        model.train()
        train(trainer, batch_size, device, optimizer, model, loss_fn)
        model.eval()
        test(tester, batch_size, device, model)
        plotMetric()
        torch.save(model.state_dict(), savePath)
    validate(model, validator, device)