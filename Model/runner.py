from Model.chrom_dataset import Chromatin_Dataset
from Model.model import *
from Model.util import  runModel, validate
import torch
from torch import nn
import timeit

def getData(chromtypes, 
            id, 
            trainLabel, 
            testLabel, 
            validLabel,
            fileLocation="./Data/220708/DATA"
        ):

    chr_train = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=trainLabel,
        file_location=fileLocation+"/TRAIN/*")

    chr_test = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=testLabel,
        file_location=fileLocation+"/HOLDOUT/*")

    chr_valid = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=validLabel,
        file_location=fileLocation+"/HOLDOUT/*")

    trainer = [chr_train]
    tester = [chr_test]
    validator = [chr_valid]

    return trainer, tester, validator   


def runner(chromtypes,  
            id="K562", 
            trainLabel="chr10-chr17", 
            testLabel="chr10", 
            validLabel="chr17", 
            epochs=2, 
            batchSize=64,
            fileLocation="./Data/220802_DATA", 
            modelType=1
        ):


    start = timeit.default_timer()

    trainer, tester, validator = getData(chromtypes,  
                            id=id, 
                            trainLabel=trainLabel, 
                            testLabel=testLabel, 
                            validLabel=validLabel,
                            fileLocation=fileLocation)

    name = "id_{}_TTV_{}_{}_{}_epoch_{}_BS_{}_FL_{}_MT_{}".format(id, trainLabel, testLabel,validLabel, epochs, batchSize, fileLocation, modelType)
    name = name.replace("/", "-")
    name = name.replace(".","")

    stop = timeit.default_timer()
    print("Reading Data time: {}".format(stop-start))

    start = timeit.default_timer()
    if modelType == 1:
        model = Chromatin_Network1(name)
    if modelType == 2:
        model = Chromatin_Network2(name)   
    if modelType == 3:
        model = Chromatin_Network3(name)   

    print(model) 

    stop = timeit.default_timer()
    print("Generating Model time: {}".format(stop-start))
    start = timeit.default_timer()


    
    runModel(   trainer,
                tester,
                validator,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss_fn=nn.BCEWithLogitsLoss(),
                batch_size=batchSize, 
                epochs=epochs
            )
    stop = timeit.default_timer()
    print('Running Model time: {}'.format(stop - start))
    



def loadModel(modelFileName, modelType):
    if modelType == 1:
        model = Chromatin_Network1("validator")
    if modelType == 2:
        model = Chromatin_Network2("validator")  
    if modelType == 3:
        model = Chromatin_Network3("validator")  

    model.load_state_dict(torch.load(modelFileName))
    return model
    

def validator(modelFilename, chromData, device):
    model = loadModel(modelFilename)
    model = model.to(device)
    model.eval()
    return validate(model, [chromData], device), chromData.labels

