from Chrom_Proj.chrom_dataset import *
from Chrom_Proj.util import *
import torch
import Chrom_Proj.visualizer as v
import numpy as np
import glob
import pdb
import os

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
    epochs = 20
    batchSize = 128

    # hostname should have type and number
    hostname = os.environ.get("HOSTNAME").split("-")
    ids = [hostname[-2]]



    # Parameters for model
    trainlabels = ["chr10-chr17"]#, "chr11-chr7", "chr12-chr8",  "chr13-chr9", "chr15-chr16"]
    otherlabels = ["chr10","chr17"]#, "chr11","chr7", "chr12","chr8",  "chr13","chr9", "chr15","chr16"]
    groupLabels = [[id,trainlabel,testlabel,validlabel] for id in ids for trainlabel in trainlabels for testlabel in otherlabels for validlabel in otherlabels]
    groupLabels = validate(groupLabels)
    
    models = [1,2,3,4,5]

    if hostname[-1] == '1':
        chromTypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
        runHomoModels(chromTypes, epochs, batchSize, groupLabels,  models, nameType="1")
    else:
        chromTypes = ["CTCF-2", "H3K4me3-2", "H3K27ac-2", "p300-2", "PolII-2"]
        runHomoModels(chromTypes, epochs, batchSize, groupLabels,  models, nameType="2")

def validate(groupLabels):
    cleanLabel = []
    for label in groupLabels:
        id = label[0]
        trainLabel = label[1]
        testLabel = label[2]
        validLabel = label[3]
        if testLabel != validLabel:
            if testLabel in trainLabel:
                if validLabel in trainLabel:
                    cleanLabel.append(label)
    return cleanLabel

def runHomoModels(chromTypes, epochs, batchSize, groupLabels,models, nameType):
    # Goes through every permutation of the variables for building and training models
    for simulation in groupLabels:
        # print(simulation)
        id = simulation[0]
        trainLabel = simulation[1]
        testLabel = simulation[2]
        validLabel = simulation[3]
        
        trainer = []
        tester = []
        validator = []

        for modelType in models:
            name = "id_{}_TTV_{}_{}_{}_epoch_{}_BS_{}_FL_{}_MT_{}_name_{}".format(id, trainLabel, testLabel,validLabel, epochs, batchSize, "./Data/220802_DATA", modelType, nameType)
            name = name.replace("/", "-")
            name = name.replace(".","")
            chr_train, chr_test, chr_valid = getData(chromTypes,  
                id=id, 
                trainLabel=trainLabel, 
                testLabel=testLabel, 
                validLabel=validLabel,
                fileLocation="./Data/220802_DATA")

            model = loadModel(modelType,name)
            print("model:{}\nid: {}\nTraining on {}\nTesting on {}\nValidating on: {}\n".format(modelType,id,chr_train, chr_test, chr_valid))
            print(name)
            print(model)
            runModel([chr_train], [chr_test], [chr_valid],
                model=model,
                optimizer= torch.optim.Adam(model.parameters(), lr=0.0001),
                loss_fn=nn.BCELoss(),
                batch_size=batchSize,
                epochs=epochs)
            

main()
