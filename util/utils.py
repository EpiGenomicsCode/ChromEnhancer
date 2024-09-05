import numpy as np
import torch.optim as optim
import tqdm
from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5, ChrNet6, ChrNet7
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve
import gc 
import torch
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
    if modelNumber == 1:
        return ChrNet1.Chromatin_Network1(name, input_size)
    elif modelNumber == 2:
        return ChrNet2.Chromatin_Network2(name, input_size)
    elif modelNumber == 3:
        return ChrNet3.Chromatin_Network3(name, input_size)
    elif modelNumber == 4:
        return ChrNet4.Chromatin_Network4(name, input_size)
    elif modelNumber == 5:
        return ChrNet5.Chromatin_Network5(name, input_size)
    elif modelNumber == 6:
        return ChrNet6.Chromatin_Network6(name, input_size)
    elif modelNumber ==7:
        return ChrNet7.Chromatin_Network7(name, [input_size, 256, 256, 256, 256, 1])
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
            outputPath: Location of output path    
        returns:
            model: The model
            loss_values: The loss values
            accuracy_values: The accuracy values
        
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # send the model to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Name: {model.name}\t Device: {device}")
    # print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)

    # initialize the values
    best_accuracy = 0

    # run the model for the specified number of epochs
    epoch = 0 
    training_accuaracy = []
    training_loss = []
    testing_accuaracy = []
    test_auROC = []
    test_auPRC = []
    for epoch in tqdm.tqdm(range(epochs), leave=False, desc="Epoch", total=epochs):
        # run the model for one epoch
        model.train()
        model, accuracy, loss = runEpoch(model, train_loader, criterion, optimizer)
        training_accuaracy.append(accuracy)
        training_loss.append(loss)

        # test the model
        model.eval()
        accuracy, auROC, auPRC = testModel(model, test_loader, criterion, outputPath, epoch, save=False)
        testing_accuaracy.append(accuracy)
        test_auROC.append(auROC)
        test_auPRC.append(auPRC)

        # check if the output folder exists
        #os.makedirs("./output/modelWeights", exist_ok=True)
        #torch.save(model.state_dict(), "./output/modelWeights/{}_epoch_{}.pt".format(model.name, epoch))

        plotAccuracy(training_accuaracy, "train_"+model.name, outputPath)
        plt.clf()
        plotAccuracy(testing_accuaracy, "test_"+model.name, outputPath)
        # save trainig and testing accuracy as a csv
        df = pd.DataFrame(training_accuaracy)
        df.to_csv(outputPath+"/accuracy/train_"+model.name+".csv")
        df = pd.DataFrame(testing_accuaracy)
        df.to_csv(outputPath+"/accuracy/test_"+model.name+".csv")
        plt.clf()
        plt.plot(test_auROC)
        plt.plot(test_auPRC)
        plt.legend(["auROC", "auPRC"])
        plt.title("auROC and auPRC")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        os.makedirs(outputPath+"/auROC_PRC", exist_ok=True)
        plt.savefig(outputPath+"/auROC_PRC/" + model.name + ".png")
        plt.clf()

        # save the aurROC and auPRC as csv
        df = pd.DataFrame(test_auROC)
        df.to_csv(outputPath+"/auROC_PRC/auROC_"+model.name+".csv")
        df = pd.DataFrame(test_auPRC)
        df.to_csv(outputPath+"/auROC_PRC/auPRC_"+model.name+".csv")
        
        # plot loss
        plt.plot(training_loss)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        os.makedirs(outputPath+"/loss", exist_ok=True)
        plt.savefig(outputPath+"/loss/" + model.name + ".png")
        plt.clf()
        # save training loss as a csv
        df = pd.DataFrame(training_loss)
        df.to_csv(outputPath+"/loss/" + model.name + ".csv")

        # Update learning rate at end of each epoch
        #scheduler.step()
                
    # test the model on the validation data
    accuracy, auROC, auPRC = testModel(model, valid_loader, criterion, outputPath, epochs, save=True)
    test_auPRC.append(auPRC)
    test_auROC.append(auROC)

    # Output final model weights
    os.makedirs(outputPath+"/modelWeights", exist_ok=True)
    torch.save(model.state_dict(), "{}/modelWeights/{}_epoch_{}.pt".format(outputPath, model.name, epochs))

    plt.clf()
    plt.plot(test_auROC)
    plt.plot(test_auPRC)
    #  plot the validation auROC and auPRC  as black dots
    plt.scatter([len(test_auROC)-1], [test_auROC[-1]], c="black")
    plt.scatter([len(test_auPRC)-1], [test_auPRC[-1]], c="black")    
    plt.legend(["auROC", "auPRC"])
    plt.title("auROC and auPRC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(outputPath+"/auROC_PRC/" + model.name + ".png")
    plt.clf()

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
    y_score = []
    y_true = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochloss = 0
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
        epochloss += loss.item()
        # save the output and label
        y_score.append(np.array(outputs.detach().cpu().numpy().tolist()).flatten())
        y_true.append(np.array(labels.detach().cpu().numpy().tolist()).flatten())


        # clear the memory
        clearCache()

    return model, accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round()), epochloss/len(train_loader)

# test the model
def testModel(model, test_loader, criterion, outputPath, epoch, save=False):
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

#    f=open(f"{outputPath}/{epoch}_inputs.tsv",'a')
    # validate the model
    for inputs, labels in tqdm.tqdm(test_loader,  desc="processing testing batches", leave=False):
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        #import pdb; pdb.set_trace()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # save the output and label
        y_score.append(np.array(outputs.detach().cpu().numpy().tolist()).flatten())
        y_true.append(np.array(labels.detach().cpu().numpy().tolist()).flatten())
#        np.savetxt(f, np.array(inputs.detach().cpu().numpy().tolist()), delimiter='\t', fmt='%.6f')
 
#    # Code to check each epoch's auPRC and auROC calculation
#    np.savetxt(f"{outputPath}/{epoch}_y_score.tsv", np.concatenate(y_score), delimiter='\t')
#    np.savetxt(f"{outputPath}/{epoch}_y_true.tsv", np.concatenate(y_true), delimiter='\t')

    acc, auROC, auPRC = calcData(model, y_score, y_true, save, outputPath)
#    import pdb; pdb.set_trace()

    return acc, auROC, auPRC

def calcData(model, y_score, y_true, save, outputPath):
    recall, precision, auPRC =   plotPRC(model, y_score, y_true, model.name, save, outputPath)
    fpr   , tpr      , auROC =   plotROC(model, y_score, y_true, model.name, save, outputPath)

    # save the results
    os.makedirs(outputPath+"/results", exist_ok=True)
    if save:
        with open("{}/results/{}.txt".format(outputPath, model.name), "w") as f:
            f.write("Recall: {}\n".format(recall))
            f.write("Precision: {}\n".format(precision))
            # write fpr and tpr as a string using join to avoid scientific notation
            f.write("FPR: {}\n".format(','.join(map(str, fpr))))
            f.write("TPR: {}\n".format(','.join(map(str, tpr))))
            f.write("auPRC: {}\n".format(auPRC))
            f.write("auROC: {}\n".format(auROC))

    return accuracy_score(np.concatenate(y_true), np.concatenate(y_score).round()), auROC, auPRC

# plot the PRC curve for the pytorch model
def plotPRC(model, y_score, y_true, name, save, outputPath):
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
    
    if save:
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
        os.makedirs(outputPath+"/PRC/", exist_ok=True)
        plt.savefig("{}/PRC/{}.png".format(outputPath, name))
        plt.clf()

    return recall, precision, auc_score

# plot the ROC curve for the pytorch model
def plotROC(model, y_score, y_true, name, save, outputPath):
    """
        Plots the ROC curve for the model

        Args:
            model: The model to plot
            test_loader: The testing data
            name: The name of the model
    """
    # plot the ROC curve
    fpr, tpr, _ = roc_curve( np.concatenate(y_true), np.concatenate(y_score))

    auc_score = round(auc(fpr, tpr),5)
    if save:
        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        # calulate the AUC score
        plt.text(0.5, 0.5, "AUC: {}".format(auc_score))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('ROC Curve')

        # check if the output folder exists
        os.makedirs(outputPath+"/ROC", exist_ok=True)
        plt.savefig("{}/ROC/{}.png".format(outputPath, name))
        plt.clf()

    return fpr, tpr, auc_score

def plotAccuracy(accuracy_values, name, outputPath):
    """
        Plots the accuracy values
    """
    plt.plot(accuracy_values)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    os.makedirs(outputPath+"/accuracy", exist_ok=True)
    plt.savefig(outputPath+"/accuracy/" + name + ".png")
    # save the data to a csv file
    df = pd.DataFrame(accuracy_values)
    df.to_csv(outputPath+"/accuracy/" + name + ".csv")
    plt.clf()
