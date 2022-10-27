from Chrom_Proj.model import *
from Chrom_Proj.util import  runModel, validate, loadModel
from Chrom_Proj.chrom_dataset import getData
import torch
from torch import nn
import timeit
import gc


    
def validator(modelFilename, chromData, device, modelType):
    """
    Loads a model into memory and runs it on  chromatin data

    Input
    ======
        modelFilename: string : relative filepath of saved model
        chromData: Data to be run
        device: string: where the data should be run 
            default: cpu
        modelType: int : type of model ()
    """
    model = loadModel(modelFilename, modelType)
    model = model.to(device)
    model.eval()
    return validate(model, [chromData], device), chromData.labels

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
    """
        Loads data into memory with model on same device, then runs data on the model 

        input
        =====
            chromtypes: list : chromtype order  
            id: string : name of chromitin
                default: "K562" 
            trainLabel: string: training data from preprocessed data 
                default: "chr10-chr17", 
            testLabel: string: testing data from preprocessed data
                default: "chr10", 
            validLabel: string: validation data from preprocessed data
                default: "chr17", 
            epochs: int: number of epochs to train each model,
                default: 2 
            batchSize: int: batch size for training
                default: 64
            fileLocation: string: location of preprocessed data
                default: "./Data/220802_DATA", 
            modelType: int: type of model
                default: 1


        returns:
        =======
            outputData: list of predictions given from validation set 
    """


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
    print("Generating model:\t{}\n\n".format(name))

    start = timeit.default_timer()
    model = loadModel(modelType, name)

    print(model) 

    stop = timeit.default_timer()
    print("Generating Model time: {}".format(stop-start))
    start = timeit.default_timer()


    
    realValid, predictedValid, model = runModel(   trainer,
                tester,
                validator,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss_fn=nn.BCELoss(),
                batch_size=batchSize, 
                epochs=epochs
                )
    stop = timeit.default_timer()
    print('Running Model time: {}'.format(stop - start))
    del model
    gc.collect() 
    torch.cuda.empty_cache()
    return realValid, predictedValid
    