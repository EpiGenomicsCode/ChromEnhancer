from Chrom_Proj.runner import runner
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
import torch
import Chrom_Proj.visualizer as v


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
    epochs = 10
    batchSize = 128

    # parameters for model
    ids = ["A549", "HepG2", "K562", "MCF7" ]
    trainLabels = ["chr10-chr17", "chr11-chr7", "chr12-chr8",  "chr13-chr9", "chr15-chr16" ]
    testLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]
    validLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]
    models = [1,2,3,4,5]

    # Goes through every permutation of the variables for building and training models
    for id in ids:
        for trainLabel in trainLabels:
            for testLabel in testLabels:
                for validLabel in validLabels:
                    for modelType in models:
                        # Compare test label and vaidation labels
                        tL = int(testLabel[testLabel.index("r")+1:]) 
                        vL = int(validLabel[trainLabel.index("r")+1:])
                        sliceLeft = int(trainLabel[trainLabel.index("r")+1:trainLabel.index("-")])
                        sliceRight = int(trainLabel[trainLabel.rindex("r")+1:])
                        
                        # We do not want to train and test on the same data
                        if tL != vL:
                            if tL == sliceLeft:
                                if vL == sliceRight:
                                    runner(chromTypes,  
                                            id=id, 
                                            trainLabel=trainLabel, 
                                            testLabel=testLabel, 
                                            validLabel=validLabel,
                                            epochs=epochs, batchSize=batchSize,
                                            fileLocation="./Data/220802_DATA", 
                                            modelType=modelType
                                            )
                                    


main()
