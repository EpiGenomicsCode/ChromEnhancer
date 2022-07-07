import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm 
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader



def clean(data, labels):
    newdata = []
    newlabel = []
    for i in zip(data.values, labels.values):
        dataValue = i[0]
        labelValue = i[1]
        if np.sum(dataValue) != 0:
            newdata.append(dataValue)
            newlabel.append(labelValue)
    return newdata, newlabel

def readfiles(chromeType,chromName, file_location):
    files = glob(file_location)
    labels = []
    data = []

    for i in files:
        if chromeType in i and chromName in i:
            if ".chromtrack" in i:
                print("Processing: {}".format(i))
                data = pd.read_csv(i, delimiter=" ")
        if chromeType in i:
            if ".label" in i:
                print("Processing: {}".format(i))
                labels = pd.read_csv(i, delimiter=" ")


    
    return clean(data, labels)

def fitSVM(epochs, train, test, valid):
   print("Running the SVM")
   supportvectormachine = svm.SVC(verbose=True, tol=1e-1,cache_size=1024, max_iter=epochs, kernel="poly", degree=7)
   for t in train:
      supportvectormachine.fit(t.data, t.labels)

   for t in test:
      print("\ntest:{}".format(supportvectormachine.score(t.data, t.labels)))

   for t in valid:
      print("\nvalidation:{}".format(supportvectormachine.score(t.data, t.labels)))
   
   return supportvectormachine

def plotPCA(train):
   x =train.data
   pca = PCA(n_components=2)
   principalComponents = pca.fit_transform(x)
   principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
   labels = pd.DataFrame(train.labels, columns=["target"])
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
   plt.savefig("output/PCA_"+train.filename+".png")

def trainModel(trainer, tester, validator, model, optimizer, loss_fn, batch_size, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # used to log training loss per epcoh
    train_losses = []
    test_losses   = []
    test_accuracy = []

   # Train the model
    for epoch in tqdm(range(epochs), position=0, leave=True):
        # Set model to train mode and train on training data
        total_train_loss = 0
        for train_loader in trainer:
            train_loader = DataLoader(train_loader, batch_size=batch_size,shuffle=True)
            for data, label in train_loader:
                # Load data appropriatly
                data, label = data.to(device), label.to(device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward Pass
                target = model(data)

                # Clean the data
                target = torch.flatten(target)

                # Calculate the Loss
                loss = loss_fn(target.to(torch.float32), label.to(torch.float32))

                # Calculate the gradient
                loss.backward()

                # Update Weight
                optimizer.step()

                # save loss
                total_train_loss += loss.item()
             # normalize for each data point
            total_train_loss/=len(train_loader)
        # normalize for each chometype
        total_train_loss/=len(trainer)

        total_test_loss = 0
        total_accuracy = 0
        for test_loader in tester:
            test_loader = DataLoader(test_loader, batch_size=batch_size,shuffle=True)

            # Set model to validation 
            model.eval()
            for test_data, test_label in test_loader:
                test_data , test_label = test_data.to(device) , test_label.to(device)

                target = model(test_data)
                test_label = test_label.unsqueeze(1)

                loss = loss_fn(target.to(torch.float32), test_label.to(torch.float32))
                total_test_loss += loss.item() 

                target = torch.round(target)
                total_accuracy += torch.sum(test_label == target).item()
            
            total_accuracy/=len(test_loader)
            total_test_loss/=len(test_loader)
        total_accuracy/=len(validator)
        total_accuracy/=batch_size
        total_test_loss /= len(validator)

        train_losses.append(total_train_loss)
        test_losses.append(total_test_loss)
        test_accuracy.append(total_accuracy)
      
        plt.plot(train_losses)
        plt.savefig("output/training_losses.png")
        plt.clf()
        plt.plot(test_losses)
        plt.savefig("output/testing_losses.png")
        plt.clf()
        torch.save(model.state_dict(), savePath)
        plt.plot(test_accuracy)
        plt.savefig("output/test_accuracy.png")
        plt.clf()

        print(f'Epoch {epoch+1} \t\t Training Loss: {total_train_loss} \t Testing Loss: {total_test_loss} \t Test Avg. Accuracy: {total_accuracy}')
      
    total_valid_loss = 0
    for valid_loader in validator:
        valid_loader = DataLoader(valid_loader, batch_size=batch_size,shuffle=True)

        # Set model to validation 
        model.eval()
        valid_loss = 0
        for valid_data, valid_label in valid_loader:
            valid_data , valid_label = valid_data.to(device) , valid_label.to(device)

            target = model(valid_data)
            valid_label = valid_label.unsqueeze(1)

            loss = loss_fn(target.to(torch.float32), valid_label.to(torch.float32))
            valid_loss += loss.item() 
        total_valid_loss+=valid_loss
    total_valid_loss /= len(validator)
    print("TOTAL VALID LOSS:{}".format(total_valid_loss))