import util.Adversarial_Observation.Adversarial_Observation as AO
from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5
from util import dataset
import shap
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import permutations
import pickle
import csv

import argparse
# add command line arguments
parser = argparse.ArgumentParser(description='Run the study')
parser.add_argument('--fileInput', type=str, help='Location of training and test data', default="./data/CELL_NETWORK/")
parser.add_argument('--modelPath', type=str, help='Location of training and test data', required=True)
parser.add_argument('--modelID', type=int, help='Location of training and test data', default=4)
# Chromatin dropout or Cellline dropout models
parser.add_argument('--parameterCHR', action='store_false', help='Run the parameter study')
parser.add_argument('--parameterCLD', action='store_false', help='Run the parameter study with cell line dropout')

# optional cellLine
parser.add_argument('--cellLine', nargs='+', help='Run the study on the cellLine', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--chromData', nargs='+', help='Run the study using the following chromatin datasets', default=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"])
parser.add_argument('--batch_size', type=int, help='Run the study on the batch size', default=2048)
parser.add_argument('--bin_size', type=int, help='How many bins to use when loading the data', default=65536)

args = parser.parse_args()

ALLCELLS = ["A549", "MCF7", "HepG2", "K562"]
ALLCHROM = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

def getValidTest(cellUse, cellHold, chrUse, fileLocation):
    print(cellUse)
    print(cellHold)
    print(chrUse)
    train, test, valid = dataset.getData(trainLabel="", testLabel="", validLabel="", chrUse=chrUse, cellUse=cellUse, cellHold=cellHold, bin_size=args.bin_size, fileLocation=fileLocation, dataTypes="-1")
    #  go through all test and only use the ones that have high confidence
    x_test = []
    y_test = []
    COUNT=0
    for data, label in test:
        if label[0] > 0.9:
            x_test.append(data)
            y_test.append(label[0])
        COUNT = COUNT + 1

    print("Total samples assessed: " + str(COUNT))
    print("Samples passing threshold: " + str(len(x_test)))
    x_test = np.array(x_test)
    y_test = np.array([[i] for i in y_test])
    test = torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return train, test, valid

def main():
    # Process commandline parameters
    cellUse = args.cellLine
    cellHoldout = list(set(ALLCELLS) - set(cellUse))
    chromData = args.chromData
    fileInput = args.fileInput

    # Load trained model
    if args.modelID == 1:
        modelarchitecture = ChrNet1.Chromatin_Network1("", 500)
    elif args.modelID == 2:
        modelarchitecture = ChrNet2.Chromatin_Network2("", 500)
    elif args.modelID == 3:
        modelarchitecture = ChrNet3.Chromatin_Network3("", 500)
    elif args.modelID == 4:
        modelarchitecture = ChrNet4.Chromatin_Network4("", 500)
    elif args.modelID == 5:
        modelarchitecture = ChrNet5.Chromatin_Network4("", 500)
    modelarchitecture.load_state_dict(torch.load(args.modelPath, map_location=torch.device('cpu')))

    # Get test data for XAI calcuations
    train, test, valid = getValidTest(cellUse=cellUse, cellHold=cellHoldout, chrUse=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"], fileLocation=fileInput)
    
    calcXAI(modelarchitecture, os.path.basename(args.modelPath) + "_" + ''.join(cellUse), train, test, valid)

def calcXAI(model, model_name, train, test, valid, data_size=1000):
    # we are using valid just to make sure the visualizations are correct
    x_train, y_train = next(iter(DataLoader(train, batch_size=data_size, shuffle=True)))
    # x_train, y_train = next(iter(DataLoader(vaild, batch_size=32, shuffle=True)))
    x_test, y_test = next(iter(DataLoader(test, batch_size=data_size, shuffle=True))) # Load the entire test set

    model = model.to('cpu')
    # send everything to cpu
    x_train = x_train.to('cpu')
    x_test = x_test.to('cpu')
    y_test = y_test.to('cpu')
    y_train = y_train.to('cpu')


    background = x_train  # Using the whole training set as background
    background = background.to(torch.float32)
    print("Initializing SHAP")
    e = shap.DeepExplainer(model, background)

    # Get SHAP values for the test set
    print("Calculating SHAP values for TEST data")
    shap_values = e.shap_values(x_test[:data_size].to(torch.float32))

    # Get the Gradient Map values for the test set
    print("Calculating gradient map values")
    gradient_values = []
    for i in range(data_size):
        gradient_map = AO.Attacks.gradient_map(x_test[i].reshape(1, 1, 500).to(torch.float32), model, (1, 500), backprop_type='guided')
        gradient_values.append(gradient_map)
    gradient_squeezed = [np.squeeze(arr) for arr in gradient_values]

    print(x_test[:data_size].numpy().shape)
    print(y_test[:data_size].numpy().shape)
    print(shap_values[:data_size].shape)
    print(np.stack(gradient_squeezed).shape)

#    # Perform K-means clustering
#    kmeans = KMeans(n_clusters=5, random_state=42)
#    kmeans.fit(shap_values[:data_size])
#    print(kmeans.labels_)
#     # Save the original data, labels, and SHAP values in a DataFrame
#     data = {
#         "Original Image": x_test[:data_size].numpy(),
#         "Original Label": y_test[:data_size].numpy(),
#         "SHAP Values": shap_values[:data_size],
#         "SHAP Kmeans": kmeans.labels_,
#         "Gradient Values": np.stack(gradient_values)
#     }
    # Save the array to a CSV file
    os.makedirs(f"./output/xai", exist_ok=True)
    np.savetxt(f"./output/xai/{model_name}_xai_orig.csv", x_test[:data_size], delimiter=',')
    np.savetxt(f"./output/xai/{model_name}_xai_shap.csv", shap_values[:data_size], delimiter=',')
    np.savetxt(f"./output/xai/{model_name}_xai_gradient.csv", np.stack(gradient_squeezed), delimiter=',')

if __name__ == "__main__":
    main()
