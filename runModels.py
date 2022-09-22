from Chrom_Proj.runner import runner
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
import torch
import Chrom_Proj.visualizer as v


# TODO train model and save validation output, send validation output to William in corresponding bed file col before the . for 
# correct validation & and PRC do multiple models on multiple data

def main():
    chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    epochs = 2
    batch_size = 128

    ids = ["A549", "HepG2", "K562", "MCF7" ]
    trainLabels = ["chr10-chr17", "chr11-chr7", "chr12-chr8",  "chr13-chr9", "chr15-chr16" ]
    testLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]
    validLabels = ["chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr7", "chr8", "chr9"]
    models = [4]

    for id in ids:
        for trainLabel in trainLabels:
            for testLabel in testLabels:
                for validLabel in validLabels:
                    for modelType in models:
                        tL = int(testLabel[testLabel.index("r")+1:]) 
                        vL = int(validLabel[trainLabel.index("r")+1:])
                        sliceLeft = int(trainLabel[trainLabel.index("r")+1:trainLabel.index("-")])
                        sliceRight = int(trainLabel[trainLabel.rindex("r")+1:])
                        
                        if tL != vL:
                            if tL == sliceLeft:
                                if vL == sliceRight:
                                    runner(chromtypes,  
                                            id=id, 
                                            trainLabel=trainLabel, 
                                            testLabel=testLabel, 
                                            validLabel=validLabel,
                                            epochs=epochs, batchSize=batch_size,
                                            fileLocation="./Data/220802_DATA", 
                                            modelType=modelType
                                            )
                                    







main()
