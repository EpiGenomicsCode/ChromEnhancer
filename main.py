import torch
import pandas as pd
from torch import nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split as rnd_splt


from Model.chrom_dataset import Chromatin_Dataset
from Model.model import Chromatin_Network
from Model.util import fitSVM, plotPCA, runModel, validate, loadModel

# TODO train model and save validation output, send validation output to William in corresponding bed file col before the . for 
# correct validation & and PRC do multiple models on multiple data

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]

    K562_chr10_17 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr10-chr17",
        file_location="./Data/220708_DATA/TRAIN/*") # 499100

    K562_chr10 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr10",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    K562_chr17 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr17",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    trainer = [K562_chr10_17]
    tester = [K562_chr10]
    validator = [K562_chr17]

    # Train the SVM
    # supportvectormachine = svm.SVC(verbose=True, tol=1e-1,cache_size=1024, kernel="poly", degree=7)
    # supportvectormachine = fitSVM(supportvectormachine, epochs, trainer, tester, validator)

    # PCA plot
    for t in trainer:
       plotPCA(t)
    for t in tester:
       plotPCA(t)
    for t in validator:
       plotPCA(t)

    
    # Detect GPU or CPU
    epochs = 5
    batch_size = 64
    learning_rate = 1e-4

    # Build the model
    model = Chromatin_Network()
    print(model)

    # Compile the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    runModel(
        trainer,
        tester,
        validator,
        model,
        optimizer,
        loss_fn,
        batch_size,
        epochs)
    
    model = loadModel()
    output = validate(model, validator, device)
    print(output[0])



main()
