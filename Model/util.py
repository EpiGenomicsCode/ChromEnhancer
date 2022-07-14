from cProfile import label
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from sklearn import preprocessing

#id="A549", chromType= ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"], label="chr10-chr17", file_location="./Data/220708_DATA/TRAIN/*"

def readfiles(id, chromType, label, file_location):
    files = glob(file_location)
    labels = []
    data = {}

    for fileType in chromType:
        filename = [i for i in files if id in i and fileType in i and "chromtrack" in i and label in i]
        assert len(filename) != 0
        print("Processing: {}".format(filename[0]))
        fileName = filename[0]

        data[fileType] = pd.read_csv(fileName, delimiter=" ")
    
    
    horizontalConcat = pd.DataFrame()
    for fileType in chromType:
        horizontalConcat = pd.concat([horizontalConcat, data[fileType]],axis=1)

    labelFileName = [i for i in files if ".label" in i and id in i and label in i]
    assert len(labelFileName) > 0
    print("Processing: {}".format(labelFileName[0])) 
    label = pd.read_csv(labelFileName[0], delimiter=" ")

    return np.array(horizontalConcat.values), np.array(label.values)

def fitSVM(supportvectormachine, epochs, train, test, valid):
   print("Running the SVM")
   for t in train:
      supportvectormachine.fit(t.data, t.labels)

   for t in test:
      print("\ntest:{}".format(supportvectormachine.score(t.data, t.labels)))

   for t in valid:
      print("\nvalidation:{}".format(supportvectormachine.score(t.data, t.labels)))
   
   return supportvectormachine

def plotPCA(chrom):
    print("Processing PCA for {}".format(chrom.filename))
    x =chrom.data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    labels = pd.DataFrame(chrom.labels, columns=["target"])
    finalDf = pd.concat([principalDf, labels], axis=1)
    finalDf.to_csv("./Data/PCA_compression.csv")
    
    colors  = ['b', 'r']
    targets = [ 0 ,  1 ]
    ax1 = plt.subplot(211)

    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA, target = 0', fontsize = 20)

    indicesToKeep = finalDf['target'] == 0
    ax1.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = 'r'
                , s = 50
                )
    
    ax2 = plt.subplot(212,sharex=ax1)
    ax2.set_title('2 component PCA, target = 1', fontsize = 20)
    ax2.set_xlabel('Principal Component 1', fontsize = 15)


    indicesToKeep = finalDf['target'] == 1
    ax2.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = 'b'
                , s = 50
                )
    plt.tight_layout()
    plt.savefig("output/PCA_"+chrom.filename+".png")

def plotData(train_losses, test_losses, test_accuracy):

    plt.plot(train_losses, label="training")
    plt.plot(test_losses, label="testing")
    plt.legend()
    plt.savefig("output/losses.png")
    plt.clf()
    
    precision = [i[0] for i in test_accuracy]
    recall = [i[1] for i in test_accuracy]
    F1 = [i[2] for i in test_accuracy]

    plt.plot(precision, label="precision")
    plt.legend()
    plt.savefig("output/precision.png")
    plt.clf()

    
    plt.plot(recall, label="recall")
    plt.legend()
    plt.savefig("output/recall.png")
    plt.clf()

    
    plt.plot(F1, label="F1")
    plt.legend()
    plt.savefig("output/F1.png")
    plt.clf()

def calcPRF(real, gen):
    truePositive = 0
    falsePositives = 0
    trueNegitive = 0
    falseNegitive = 0
    epsilon = 1e-8
    for i in zip(real, gen):
        if i[0] == 1:
            if i[1] == 1:
                truePositive+=1
            else:
                falseNegitive+=1
        else:
            if i[1]==1:
                falsePositives+=1
            else:
                trueNegitive+=1

    # epsilon added to avoid divide by zero error
    precision = truePositive/(truePositive+falsePositives+epsilon)
    recall = truePositive/(truePositive+falseNegitive+epsilon)
    F1 = 2*(precision*recall)/(epsilon+precision+recall)

    return precision, recall,F1
 
def trainModel(trainer, tester, validator, model, optimizer, loss_fn, batch_size, epochs):
    savePath = "output/model.pt"


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n\n============Training on: {}===========\n".format(device))
    model = model.to(device)

    # used to log training loss per epcoh
    train_losses = []
    test_losses   = []
    test_accuracy = []

   # Train the model
    for epoch in range(epochs):
        # Set model to train mode and train on training data
        total_train_loss = 0
        model.train()
        for train_loader in trainer:
            train_loader = DataLoader(train_loader, batch_size=batch_size,shuffle=True)
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
                loss = loss_fn(target.to(torch.float32), label.to(torch.float32))
                # Calculate the gradient
                loss.backward()
                # Update Weight
                optimizer.step()
                # save loss
                total_train_loss += loss.item()

        total_test_loss = 0
        precision = 0
        recall = 0
        F1 = 0
        model.eval()
        for test_loader in tester:
            test_loader = DataLoader(test_loader, batch_size=batch_size,shuffle=True)
            for test_data, test_label in tqdm(test_loader, desc="testing"):
                test_data , test_label = test_data.to(device) , test_label.to(device)

                target = model(test_data)
                target = torch.flatten(target)
                test_label = torch.flatten(test_label)
                
                loss = loss_fn(target.to(torch.float32), test_label.to(torch.float32))
                total_test_loss += loss.item() 
                p, r, f = calcPRF(test_label, target)
                precision += p
                recall += r
                F1 += f

        
        
        train_losses.append(total_train_loss)
        test_losses.append(total_test_loss)
        test_accuracy.append([precision, recall, F1])

        plotData(train_losses, test_losses, test_accuracy)
        torch.save(model.state_dict(), savePath)

        print("Epoch:{}\tTraining Loss:{:.3f}\tTesting Loss:{:.3f}\tTest Prec:{:.3f}\tTest recall:{:.3f}\tTest F1: {:.3f}".format(epoch+1,total_train_loss, total_test_loss,precision,recall,F1))
    
    precision = 0
    recall = 0
    F1 =0
    total_PRC = 0
    for valid_loader in validator:
        valid_loader = DataLoader(valid_loader, batch_size=batch_size,shuffle=True)

        # Set model to validation 
        model.eval()
        valid_loss = 0
        for valid_data, valid_label in valid_loader:
            valid_data , valid_label = valid_data.to(device) , valid_label.to(device)

            target = model(valid_data)
            target = torch.flatten(target)
            valid_label = torch.flatten(valid_label)

            p,r,f = calcPRF(valid_label, target)
            precision += p
            recall += r
            F1 += f
        
    print("Validation Precision:{}\tRecall:{}\tF1:{}".format(precision, recall, F1))
