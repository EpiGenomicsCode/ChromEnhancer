import numpy as np
import torch.optim as optim
import tqdm
from util.model import *
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def plotLoss(loss_values, name):
    """
        Plots the loss values
    """
    # convert loss_values from a  list of tensors to a list of floats
    loss_values = [i.cpu().item() for i in loss_values]

    plt.plot(loss_values)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs("./output/loss", exist_ok=True)
    plt.savefig("./output/loss/" + name + ".png")
    plt.clf()

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

def loadModel(modelNumber, name=""):
    """
        Loads the model based on the model number   
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if modelNumber == 0:
        return Chromatin_Network1(name)
    elif modelNumber == 1:
        return Chromatin_Network2(name)
    elif modelNumber == 2:
        return Chromatin_Network3(name)
    elif modelNumber == 3:
        return Chromatin_Network4(name)
    elif modelNumber == 4:
        return Chromatin_Network5(name)
    else:
        raise Exception("Invalid model number")

def runHomoModel(model, train_loader, test_loader, valid_loader, epochs, criterion, optimizer):
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
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize the values
    loss_values = []
    accuracy_values = []
    best_accuracy = 0

    # run the model for the specified number of epochs
    for epoch in tqdm.tqdm(range(epochs)):
        # run the model for one epoch
        model, loss, accuracy = runEpoch(model, train_loader, criterion, optimizer)
        # save the loss and accuracy
        loss_values.append(loss)
        accuracy_values.append(accuracy)

        # test the model
        accuracy = testModel(model, test_loader, criterion)
        # save the model if it is the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # check if the output folder exists
            os.makedirs("./output/modelWeights", exist_ok=True)
            torch.save(model.state_dict(), "./output/modelWeights/{}.pt".format(model.name))
        
    # test the model on the validation data
    accuracy = testModel(model, valid_loader, criterion)
    print("Validation Accuracy: ", accuracy)

    # return the model, loss values, and accuracy values
    return model, loss_values, accuracy_values

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
            loss: The loss
            accuracy: The accuracy
    """
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # initialize the values
    loss = 0
    accuracy = 0
    count = 0

    # run the model for one epoch with tqdm
    for inputs, labels in train_loader:
        # send the data to the gpu if available
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate the accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        count += labels.size(0)
    
    # return the model, loss, and accuracy
    return model, loss, accuracy/count

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
    model.eval()
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize the values
    loss = 0
    accuracy = 0
    count = 0

    # test the model
    for inputs, labels in test_loader:
        # send the data to the gpu if available

        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # calculate the accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        count += labels.size(0)
    
    # return the accuracy
    return accuracy/count

# plot the PRC curve for the pytorch model
def plotPRC(model, test_loader, name):
    """
        Plots the PRC curve for the model

        Args:
            model: The model to plot
            test_loader: The testing data
            name: The name of the model
    """
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize the values
    y_score = []
    y_true = []

    # get the predictions and labels
    for inputs, labels in test_loader:
        # send the data to the gpu if available
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)

        # get the predictions
        _, predicted = torch.max(outputs.data, 1)
        y_score.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    
    # plot the PRC curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    # calulate the AUC score
    auc_score = auc(recall, precision)
    plt.text(0.5, 0.5, "AUC: {}".format(auc_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    # check if the output folder exists
    os.makedirs("./output/PRC", exist_ok=True)
    # save the y_true, y_score, recall and precision as a csv file using pandas
    data = {'y_true': y_true, 'y_score': y_score, 'recall': recall, 'precision': precision}
    for key in data:
        df = pd.DataFrame(data[key])
        df.to_csv("./output/PRC/{}_{}.csv".format(key, name), index=False)

    plt.savefig("./output/PRC/{}.png".format(name))
    plt.clf()

# plot the ROC curve for the pytorch model
def plotROC(model, test_loader, name):
    """
        Plots the ROC curve for the model

        Args:
            model: The model to plot
            test_loader: The testing data
            name: The name of the model
    """
    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize the values
    y_score = []
    y_true = []

    # get the predictions and labels
    for inputs, labels in test_loader:
        # send the data to the gpu if available
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)

        # get the predictions
        _, predicted = torch.max(outputs.data, 1)
        y_score.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    
    # plot the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    # calulate the AUC score
    auc_score = auc(fpr, tpr)
    plt.text(0.5, 0.5, "AUC: {}".format(auc_score))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')

    # check if the output folder exists
    os.makedirs("./output/ROC", exist_ok=True)

    # save the y_true, y_score, fpr and tpr as a csv
    data = {'y_true': y_true, 'y_score': y_score, 'fpr': fpr, 'tpr': tpr}
    for key in data:
        df = pd.DataFrame(data[key])
        df.to_csv("./output/ROC/{}_{}.csv".format(key, name), index=False)

    plt.savefig("./output/ROC/{}.png".format(name))
    plt.clf()


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
        

    



    
