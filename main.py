import torch
import pandas as pd
from torch import nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split as rnd_splt


from Model.chrom_dataset import Chromatin_Dataset
from Model.model import Chromatin_Network
from Model.util import fitSVM, plotPCA, trainModel


def main():

    chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]

    K562_chr10_17 = Chromatin_Dataset(
        id="K562",
        chromType=[
            "CTCF-1",
            "H3K4me3-1",
            "H3K27ac-1",
            "p300-1",
            "PolII-1"],
        label="chr10-chr17",
        file_location="./Data/220708_DATA/TRAIN/*")

    K562_chr10 = Chromatin_Dataset(
        id="K562",
        chromType=[
            "CTCF-1",
            "H3K4me3-1",
            "H3K27ac-1",
            "p300-1",
            "PolII-1"],
        label="chr10",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    K562_chr17 = Chromatin_Dataset(
        id="K562",
        chromType=[
            "CTCF-1",
            "H3K4me3-1",
            "H3K27ac-1",
            "p300-1",
            "PolII-1"],
        label="chr17",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    trainer = [K562_chr10_17]
    tester = [K562_chr10]
    validator = [K562_chr17]

    # Train the SVM
    # supportvectormachine = svm.SVC(verbose=True, tol=1e-1,cache_size=1024, max_iter=epochs, kernel="poly", degree=7)
    # supportvectormachine = fitSVM(supportvectormachine, epochs, trainer, tester, validator)

    # # PCA plot
    # for t in trainer:
    #    plotPCA(t)
    # for t in tester:
    #    plotPCA(t)
    # for t in validator:
    #    plotPCA(t)

    # paramets

    # Detect GPU or CPU
    epochs = 3
    batch_size = 256
    learning_rate = 1e-3
    inputSize = 500

    # Build the model
    model = Chromatin_Network(input_shape=inputSize)
    print(model)

    # Compile the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    trainModel(
        trainer,
        tester,
        validator,
        model,
        optimizer,
        loss_fn,
        batch_size,
        epochs)


main()
