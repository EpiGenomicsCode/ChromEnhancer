from Chrom_Proj.util import runModel
from Chrom_Proj.runner import  getData
from Chrom_Proj.model import *
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
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
    ids = ["A549", "HepG2", "K562", "MCF7" ]

    #=====================================
    # load the best model
    model = Chromatin_Network1("bestHeteroChrom1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossfn= nn.BCEWithLogitsLoss()
    batchSize = 256

    chromTypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    runHeteroModels(chromTypes,epochs,batchSize,ids, model, optimizer, lossfn)

def runHeteroModels(chromTypes,epochs,batchSize,ids, model, optimizer, lossfn):
    data = []
    for id in ids:
        print("id: {}".format(id))
        # for each ID
        trainer = []
        tester = []
        validator = []
        # Load in all of its data
        trainData, testData, validData = getData(chromTypes,id, '','','', fileLocation="./Data/220803_CelllineDATA/")

        trainer.append(trainData[0])
        tester.append(testData[0])
        validator.append(validData[0])
        
        print("training on {}".format(id))
        
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
        # del model
        # gc.collect() 
        # torch.cuda.empty_cache()
    print(data)
                          

    
main()
