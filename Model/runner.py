from Model.chrom_dataset import Chromatin_Dataset
from Model.model import Chromatin_Network
from Model.util import  runModel, validate
import torch
from torch import nn


def getData(chromtypes, id, trainLabel, testLabel, validLabel):
    chr10_17 = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=trainLabel,
        file_location="./Data/220708_DATA/TRAIN/*")

    chr10 = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=testLabel,
        file_location="./Data/220708_DATA/HOLDOUT/*")

    chr17 = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=validLabel,
        file_location="./Data/220708_DATA/HOLDOUT/*")

    trainer = [chr10_17]
    tester = [chr10]
    validator = [chr17]

    return trainer, tester, validator   


def runner(chromtypes,  id="K562", trainLabel="chr10-chr17", testLabel="chr10", validLabel="chr17", epochs=2, batchSize=64):
    model = Chromatin_Network(id)
    trainer, tester, validator = getData(chromtypes,  
                            id=id, 
                            trainLabel=trainLabel, 
                            testLabel=testLabel, 
                            validLabel=validLabel)
    
    runModel(   trainer,
                tester,
                validator,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss_fn=nn.BCEWithLogitsLoss(),
                batch_size=batchSize, 
                epochs=epochs
            )


def loadModel(modelFileName):
    model = Chromatin_Network("validator")
    model.load_state_dict(torch.load(modelFileName))
    return model
    

def validator(modelFilename, chromData, device):
    model = loadModel(modelFilename)
    model = model.to(device)
    model.eval()
    return validate(model, [chromData], device), chromData.labels

