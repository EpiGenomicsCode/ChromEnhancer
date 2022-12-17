from Chrom_Proj.chrom_dataset import *
from Chrom_Proj.util import *
import torch
import Chrom_Proj.visualizer as v
import numpy as np
import glob
import pdb

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
    epochs = 10
    batchSize = 128

    # Parameters for model
    ids = ["A549" ]
    labels = ["chr10-chr17", "chr11-chr7", "chr12-chr8",  "chr13-chr9", "chr15-chr16"]
    groupLabels = [[h,i] for h in ids for i in labels]
    models = [1]

    chromTypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    runHomoModels(chromTypes, epochs, batchSize, groupLabels,  models, nameType="1")

    # chromTypes = ["CTCF-2", "H3K4me3-2", "H3K27ac-2", "p300-2", "PolII-2"]
    # runHomoModels(chromTypes, epochs, batchSize, ids, trainLabels,  models,nameType="2")


def runHomoModels(chromTypes, epochs, batchSize, groupLabels,models, nameType):
    # Goes through every permutation of the variables for building and training models
    for simulation in groupLabels:
        chromType = groupLabels[0]
        trainLabel = simulation[1]
        temp = simulation[1].split('-')
        trainValue = temp[0]
        valValue = temp[1]
        runs = [[trainValue, valValue], [valValue, trainValue]]
        # print(chromType, runs)
        for run in runs:
            testLabel = run[0]
            validLabel = run[1]
            trainer = []
            tester = []
            validator = []
            print("training on {}, testing on {}, valid on {}".format(trainLabel, testLabel, validLabel))
            chr_train, chr_test, chr_valid = getData(chromTypes,  
                id=id, 
                trainLabel=trainLabel, 
                testLabel=testLabel, 
                validLabel=validLabel,
                fileLocation="./Data/220802_DATA")
            trainer.append(chr_train)
            tester.append(chr_test)
            validator.append(chr_valid)

            for modelType in models:
                name = "id_{}_TTV_{}_{}_{}_epoch_{}_BS_{}_FL_{}_MT_{}_name_{}".format(id, trainLabel, testLabel,validLabel, epochs, batchSize, "./Data/220802_DATA", modelType, nameType)
                name = name.replace("/", "-")
                name = name.replace(".","")
                print(name)
                model = loadModel(modelType,name)
                if not "./output/model_weight_bias/model_"+name+".pt" in glob.glob("./output/model_weight_bias/*"):
                    print("model:{}\nid: {}\nTraining on {}\nTesting on {}\nValidating on: {}\n".format(modelType,id,trainLabel, testLabel, validLabel))
                    print(model)
                    runModel(trainer,
                        tester,
                        validator,
                        model=model,
                        optimizer= torch.optim.Adam(model.parameters(), lr=0.0001),
                        loss_fn=nn.BCELoss(),
                        batch_size=batchSize,
                        epochs=epochs)
        
main()
