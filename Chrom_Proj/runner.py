from Chrom_Proj.chrom_dataset import Chromatin_Dataset
from Chrom_Proj.model import *
from Chrom_Proj.util import  runModel, validate
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
    """
    Returns the training, testing and validation data based on the input

    Input:
        chromtypes: List of String that represent the order of the chromatine types 
            (ex: ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"])
        
        id: String contaning the whole Chromatine Cell identification 
            (ex: "A549")
        
        trainLabel: String containing the training Label 
            (ex: "chr10-chr17")
        
        testLabel: String containing the test Label 
            (ex: "chr10")
        
        validLabel: String contatining the validation labels 
            (ex: "chr11")
        
        fileLocation: Relative file path for where the files are being saved 
            (ex: ./Data/220708/DATA)

    Return:
        trainer: list of the training data
        
        tester: list of the testing data
        
        validator: list of the validation data
    """
    
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


def loadModel(modelFileName, modelType):
    """
    Loads the models architecture from a specific file location

    Inputs:
        modelFileName: filename of the model
        modelType: the architechture of the model
    
    Returns:
        model: Pytorch Model with architecture
    """
    if modelType == 1:
        model = Chromatin_Network1("validator")
    if modelType == 2:
        model = Chromatin_Network2("validator")  
    if modelType == 3:
        model = Chromatin_Network3("validator")  
    if modelType == 4:
        model = Chromatin_Network4("validator")  
    if modelType == 5:
        model = Chromatin_Network5("validator")  
    if modelType == 6:
        model = Chromatin_Network6("validator")  

    model.load_state_dict(torch.load(modelFileName))
    
    return model
    
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
    if modelType == 1:
        model = Chromatin_Network1(name)
    if modelType == 2:
        model = Chromatin_Network2(name)   
    if modelType == 3:
        model = Chromatin_Network3(name)   
    if modelType == 4:
        model = Chromatin_Network4(name)   
    if modelType == 5:
        model = Chromatin_Network5(name)   
    if modelType == 6:
        model = Chromatin_Network6(name)   

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
    