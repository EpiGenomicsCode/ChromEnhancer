from Chrom_Proj.util import runModel
from Chrom_Proj.runner import  getData
from Chrom_Proj.model import *
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
import torch
from torch import nn
import Chrom_Proj.visualizer as v
import numpy as np
import pdb

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

    # DO NOT TOUCH THIS
    chromTypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    
    # Variables
    epochs = 10//4
    batchSize = 128

    # parameters for model
    ids = ["A549", "HepG2", "K562", "MCF7" ]
    trainLabels = ["chr10-chr17", "chr11-chr7", "chr12-chr8",  "chr13-chr9", "chr15-chr16" ]
    testLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]
    validLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]


    #=====================================
    # load the best model
    model = Chromatin_Network1("bestHeteroChrom1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossfn= nn.BCEWithLogitsLoss()
    batchSize = 256
    data = []
    for id in ids:
        # for each ID
        trainer = []
        tester = []
        validator = []
        # Load in all of its data
        for trainLabel in trainLabels:
            for testLabel  in testLabels:
                for validLabel in validLabels:
                    tL = int(testLabel[testLabel.index("r")+1:]) 
                    vL = int(validLabel[trainLabel.index("r")+1:])
                    sliceLeft = int(trainLabel[trainLabel.index("r")+1:trainLabel.index("-")])
                    sliceRight = int(trainLabel[trainLabel.rindex("r")+1:])
                    
                    # We do not want to train and test on the same data
                    if tL != vL:
                        if tL == sliceLeft:
                            if vL == sliceRight:
                                trainData, testData, validData = getData(chromTypes,id, trainLabel,testLabel,validLabel)
                                trainer.append(trainData[0])
                                tester.append(testData[0])
                                validator.append(validData[0])
        realValid, predictedValid, model =  runModel(  trainer,
                                                tester,
                                                validator,
                                                model,
                                                optimizer,
                                                lossfn,
                                                batchSize,
                                                epochs)
        rmse = np.sqrt(np.mean((np.subtract(predictedValid.tolist(),realValid.flatten().tolist()))**2))                 
        print("\t\tValidation for {} RMSE: {}".format(id, rmse)) 
        data.append((rmse, id))                             

    
main()
