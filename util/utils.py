import numpy as np
import torch.optim as optim
import tqdm
from util.model import *
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_curve, auc, accuracy_score
import gc 
import seaborn as sns
import datetime

def plotAccuracy(accuracy_values, name):
    """
        Plots the accuracy values
    """
    plt.plot(accuracy_values)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    os.makedirs("./output/accuracy", exist_ok=True)
    plt.savefig("./output/accuracy/" + name + ".png")
    plt.clf()

def loadModel(modelNumber, name="", input_size=500):
    """
        Loads the model based on the model number   
    """
    if modelNumber == 1:
        return Chromatin_Network1(name, input_size)
    elif modelNumber == 2:
        return Chromatin_Network2(name, input_size)
    elif modelNumber == 3:
        return Chromatin_Network3(name, input_size)
    elif modelNumber == 4:
        return Chromatin_Network4(name, input_size)
    elif modelNumber == 5:
        return Chromatin_Network5(name)
    elif modelNumber == 6:
        return Chromatin_Network6(name)
    else:
        raise Exception("Invalid model number")

# clear the cache and gpu memory
def clearCache():
    torch.cuda.empty_cache()
    gc.collect()

def runHomoModel(model, train_loader, test_loader, valid_loader, epochs):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)

    # initialize the values
    best_accuracy = 0

    # run the model for the specified number of epochs
    epoch = 0 
    training_accuaracy = []
    valid_accuaracy = []
    for epoch in tqdm.tqdm(range(epochs), leave=True, desc="Epoch", total=epochs):
        # run the model for one epoch
        model.train()
        model, accuracy = runEpoch(model, train_loader, criterion, optimizer)
        training_accuaracy.append(accuracy)

        # test the model
        model.eval()
        accuracy = testModel(model, valid_loader, criterion)
        valid_accuaracy.append(accuracy)

        # save the model if it is the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # check if the output folder exists
            os.makedirs("./output/modelWeights", exist_ok=True)
            torch.save(model.state_dict(), "./output/modelWeights/{}.pt".format(model.name))
        plotAccuracy(training_accuaracy, "train_"+model.name)
        plotAccuracy(valid_accuaracy, "valid_"+model.name)
        
    # test the model on the validation data
    accuracy = testModel(model, valid_loader, criterion)
    print("Validation Accuracy: ", accuracy)

    # return the model, loss values, and accuracy values
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
   
   
    # initialize the values
    loss = 0
    accuracy = 0
    count = 0
    y_score = []
    y_true = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # run the model for one epoch with tqdm
    for inputs, labels in tqdm.tqdm(train_loader, desc="processing training batches", leave=False):
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # save the output and label
        y_score.append(np.array(outputs.detach().cpu().numpy().tolist()).flatten())
        y_true.append(np.array(labels.detach().cpu().numpy().tolist()).flatten())


        # clear the memory
        clearCache()

    return model, accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round())

# test the model
def testModel(model, test_loader, criterion):
    """
        Tests the model

        Args:
            model: The model to test
            test_loader: The testing data
            criterion: The loss function
        
        returns:
            accuracy: The accuracy
    """
    # initialize the values
    loss = 0
    accuracy = 0
    count = 0
    y_score = []
    y_true = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # validate the model
    for inputs, labels in tqdm.tqdm(test_loader,  desc="processing testing batches", leave=False):
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # save the output and label
        y_score.append(np.array(outputs.detach().cpu().numpy().tolist()).flatten())
        y_true.append(np.array(labels.detach().cpu().numpy().tolist()).flatten())

    recall, precision, auPRC =   plotPRC(model, y_score, y_true, model.name)
    fpr   , tpr      , auROC =   plotROC(model, y_score, y_true, model.name)   

    # save the results
    os.makedirs("./output/results", exist_ok=True)

    with open("./output/results/{}.txt".format(model.name), "w") as f:
        f.write("Recall: {}\n".format(recall))
        f.write("Precision: {}\n".format(precision))
        # write fpr and tpr as a string using join to avoid scientific notation
        f.write("FPR: {}\n".format(','.join(map(str, fpr))))
        f.write("TPR: {}\n".format(','.join(map(str, tpr))))
        f.write("auPRC: {}\n".format(auPRC))
        f.write("auROC: {}\n".format(auROC))

    return accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round())

# plot the PRC curve for the pytorch model
def plotPRC(model, y_score, y_true, name):
    """
        Plots the PRC curve for the model

        Args:
            model: The model to plot
            test_loader: The testing data
            name: The name of the model
    """
    # plot the PRC curve
    
    precision, recall, _ = precision_recall_curve(np.concatenate(y_true), np.concatenate(y_score))

    auc_score = round(auc(recall, precision),5)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    # calulate the AUC score
    plt.text(0.5, 0.5, "AUC: {}".format(auc_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')

    # check if the output folder exists
    os.makedirs("./output/PRC/", exist_ok=True)
    plt.savefig("./output/PRC/{}.png".format(name))
    plt.clf()

    

    return recall, precision, auc_score


# plot the ROC curve for the pytorch model
def plotROC(model, y_score, y_true, name):
    """
        Plots the ROC curve for the model

        Args:
            model: The model to plot
            test_loader: The testing data
            name: The name of the model
    """
    # plot the ROC curve
    fpr, tpr, _ = roc_curve( np.concatenate(y_true), np.concatenate(y_score))
    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    # calulate the AUC score
    auc_score = round(auc(fpr, tpr),5)
    plt.text(0.5, 0.5, "AUC: {}".format(auc_score))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')

    # check if the output folder exists
    os.makedirs("./output/ROC", exist_ok=True)
    plt.savefig("./output/ROC/{}.png".format(name))
    plt.clf()

    return fpr, tpr, auc_score

def clusterParticles(particles, numClusters):
    """
        Clusters the particles using hierarchical clustering

        Args:
            particles: The particles to cluster
            numClusters: The number of clusters to use

        Returns:
            The clusters

    """
    # cluster the particles
    clustering = AgglomerativeClustering(n_clusters=numClusters).fit(particles)
    # get the clusters
    clusters = clustering.labels_
    
    return clusters

def plotClusters(clusters, particles, name):
    """
        plot a heatmap of the clusters positions
        
    """
    # get the number of clusters
    numClusters = len(np.unique(clusters))
    data = []
    for row in range(numClusters):
        # get the cluster
        clusterData = np.mean(particles[clusters == row], 0)
        clusterSections = [
                                sum(clusterData[:100]),
                                sum(clusterData[100:200]),
                                sum(clusterData[200:300]),
                                sum(clusterData[300:400]),
                                sum(clusterData[400:])
                          ]
        data.append(clusterSections)

    # plot the heatmap
    x = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    sns.heatmap(data, cmap="YlGnBu", xticklabels=x, yticklabels=["C"+str(i) for i in range(numClusters)])
    # check if the dir exists and save the heatmap
    os.makedirs("./output/heatmap", exist_ok=True)
    plt.savefig("./output/heatmap/{}.png".format(name))

    plt.clf()
        

