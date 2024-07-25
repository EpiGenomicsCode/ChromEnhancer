import sys, os
import pandas as pd
import numpy as np
import h5py

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5, ChrNet6
from util import dataset

import argparse
# add command line arguments
parser = argparse.ArgumentParser(description='Run the study')
parser.add_argument('--fileInput', type=str, help='Location of training and test data', default="./data/CELL_NETWORK/")
parser.add_argument('--fileOutput', type=str, help='Name of output file', default="predictions.tab")
parser.add_argument('--modelPath', type=str, help='Location of trained model', required=True)
parser.add_argument('--modelType', type=int, help='Model number', required=True)
parser.add_argument('--dataType', type=str, help='Data replicate ID', default="-1")

# optional cellLine
parser.add_argument('--cellLine', nargs='+', help='Run the study on the cellLine', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--chromData', nargs='+', help='Run the study using the following chromatin datasets', default=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"])

args = parser.parse_args()

ALLCELLS = ["A549", "MCF7", "HepG2", "K562"]
ALLCHROM = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"]

def getData(model, cellUse, cellHold, chrUse, fileLocation, output_file, dataType):
    # Get dataset
    train, test, valid = dataset.getData(trainLabel="", testLabel="", validLabel="", chrUse=chrUse, cellUse=cellUse, cellHold=cellHold, bin_size=2048, fileLocation=fileLocation, dataTypes=dataType)

    #  go through all test and only use the ones that have high confidence
    x_test = []
    COUNT=0
    for data, label in tqdm(test, desc="Processing", unit="batch"):
        x_test.append(data)
        COUNT = COUNT + 1
        if COUNT == 10000:
            testData = torch.utils.data.TensorDataset(torch.tensor(np.array(x_test)))
            predictions = getPredictions(model, testData, output_file)
            COUNT = 0
            x_test = []
#            break

    testData = torch.utils.data.TensorDataset(torch.tensor(np.array(x_test)))
    getPredictions(model, testData, output_file)
    return test

def getPredictions(model, test, output_file):
    x_test = next(iter(DataLoader(test, batch_size=3000000, shuffle=False))) # Load the entire test set
    # send everything to cpu
    x_test = x_test[0].to('cpu')

    predictions = []
    with torch.no_grad():
        outputs= model(x_test.to(torch.float32))
        predictions.append(outputs.detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    predictions = pd.DataFrame(predictions)
    # Append predictions to the output file
    with open(output_file, 'a') as f:
        predictions.to_csv(f, header=f.tell()==0, index=False)  # Write header only if file is empty

    #predictions.to_csv("prediction.csv")
    return predictions

def loadModel(modelNumber, name="", input_size=500):
    """
    Loads the model based on the model number   
    """
    modelNumber = int(modelNumber)
    
    model_classes = {
        1: ChrNet1.Chromatin_Network1,
        2: ChrNet2.Chromatin_Network2,
        3: ChrNet3.Chromatin_Network3,
        4: ChrNet4.Chromatin_Network4,
        5: ChrNet5.Chromatin_Network5,
        6: ChrNet6.Chromatin_Network6
    }

    if modelNumber in model_classes:
        return model_classes[modelNumber](name, input_size)
    else:
        raise Exception("Invalid model number {}".format(modelNumber))

if __name__ == "__main__":
    # Process commandline parameters
    fileInput = args.fileInput
    fileOutput = args.fileOutput
    modelPath = args.modelPath
    modelType = args.modelType
    dataType = args.dataType

    cellUse = args.cellLine
    cellHoldout = list(set(ALLCELLS) - set(cellUse))
    chromData = args.chromData

    print("Calculating predictions on: " + str(cellHoldout))

    # Load model
    print("Loading model: " + modelPath)
    model = loadModel(modelType, "", 800)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval()
    print("Model loaded\n")

    # Initialize output file
    with open(fileOutput, 'w') as f:
        f.write(str(cellHoldout) + "\n")

    test = getData(model, cellUse=cellUse, cellHold=cellHoldout, chrUse=chromData, fileLocation=fileInput, output_file=fileOutput, dataType=dataType)
