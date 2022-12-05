from Chrom_Proj.util import *
from Chrom_Proj.model import *
from Chrom_Proj.chrom_dataset import *
import torch
from torch import nn
import Chrom_Proj.visualizer as v
import numpy as np
import pdb
import gc

# TODO train model and save validation output, send validation output to William in corresponding bed file col before the . for 
# correct validation & and PRC do multiple models on multiple data

def main():
    """ 
    This is the main function to build and train all models

    Variables:
        chromTypes: the format that we want to have all inputs follow for chromatine layers
        epochs: number of epochs to run each model
        batchSize: the batch size used for training
        ids: types of chromatine to look at
        trainLabels: the sections of the chromatine we want to train on
        testLabels: the sections of the chromatine we want to test on
        validLabels: the sections of the chromatine we want to validate on
        models: list of the model types we want to builld
    """

    
    # Variables
    epochs = 1
    batchSize = 128

    # parameters for model
    #=====================================
    runHeteroModels(batchSize, epochs)

def runHeteroModels(batchSize, epochs):
    # Goes through every permutation of the variables for building and training models
    trainer = []
    tester = []
    validator = []

    ids = ["A549", "HepG2", "K562", "MCF7" ]
    chromTypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    best = 2
    model = loadModel(best, "best")

    for id in ids:
        chr_train, chr_test, chr_valid  = getData(chromTypes, 
            id, 
            '', 
            '', 
            '',
            fileLocation="./Data/220803_CelllineDATA",
            drop="PolII-1"
        )
        trainer.append(chr_train)
        tester.append(chr_test)
        validator.append(chr_valid)
    
    runModel(trainer,
        tester,
        validator,
        model=model,
        optimizer= torch.optim.Adam(model.parameters(), lr=0.0001),
        loss_fn=nn.BCELoss(),
        batch_size=batchSize,
        epochs=epochs)
                          

    
main()
