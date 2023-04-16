from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import itertools
import os
import pandas as pd
import pdb
import subprocess
import gc
import pdb
import pandas as pd
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from multiprocessing import Pool


class Chromatin_Dataset(Dataset):
    def __init__(
        self,
        cellLinesDrop,
        chromTypesDrop,
        label,
        file_location,
        dataUse,
        bin_size=10,
        dataTypes="-1",
    ):
        super(Dataset, self).__init__()

        self.cellLines = ["A549", "MCF7", "HepG2", "K562"]
        self.chromatine = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

        # Saving the parameters
        self.cellLinesDrop = cellLinesDrop
        self.chrDrop = chromTypesDrop
        self.label = label
        self.file_location = file_location
        self.dataUse = dataUse
        self.dataTypes = dataTypes

        # initialize bining variables
        self.count = -1
        self.start_index = 0
        self.bin_size = max(bin_size//(len(self.cellLines)-len(self.cellLinesDrop)), 1)
        self.end_index =0

        # initialize data and label variables
        self.data = None
        self.labels = None
        self.labelFile = None

        # Load in file paths for data and label files
        self.dataFiles = np.array(self.getDataFiles())
        self.labelFiles = np.array(self.getLabelFile())

        self.length = self.getNumSamples()
        # print all datafiles as an indented string

        print(f"Usage: {self.dataUse}")
        print(f"length: {self.length}")
        for datafile in self.dataFiles:
            for labelfile in self.labelFiles:
                print("\t",labelfile)
                for chr in datafile:
                    print("\t\t", chr[1], "\t", chr[0])
        print("=================================\n")
       


    def getDataFiles(self):
        datafiles = []

        for cellLine in self.cellLines:
            cellLineFiles = []
            for chr in self.chromatine:
                if cellLine not in self.cellLinesDrop:
                    files = glob(
                        f"{self.file_location}/*{cellLine}*{self.label}*{chr}*{self.dataTypes}*"
                    )
                    if chr not in self.chrDrop:
                        cellLineFiles.append([1, files[0]])
                    else:
                        cellLineFiles.append([0, files[0]])
            if len(cellLineFiles) != 0:
                datafiles.append(cellLineFiles)

        return datafiles

    def getLabelFile(self):
        labelNames = []
        for cellLine in self.cellLines:
            if cellLine not in self.cellLinesDrop:
                fileFormat = f"{self.file_location}/*{cellLine}*{self.label}*.label*"
                files = glob(fileFormat)
                if self.dataUse == "train":
                    labelName = [i for i in files if i and not "Leniant" in i and not "Stringent" in i ][0]
                if self.dataUse == "test":
                    labelName = [f for f in files if "Stringent" in f][0]
                if self.dataUse == "valid":
                    labelName = [f for f in files if "Lenient" in f][0]
                if len(labelName) != 0:
                    labelNames.append(labelName)
      
        return labelNames

    def getNumSamples(self):
        numSamples = []
        for labelFile in self.labelFiles:
            numSamples.append(sum(1 for line in open(labelFile)))
        return sum(numSamples)

    def __len__(self):
        return self.length
    
    def loadbin(self):
        self.start_index = self.end_index
        self.end_index = min(self.end_index + self.bin_size, self.length)

        # Load data for this bin
        batch_data_list = []
        batch_label_list = []

        with Pool(processes=len(self.dataFiles)) as pool:
            args = [
                (data_file, label_file, self.start_index, self.bin_size)
                for data_file, label_file in zip(self.dataFiles, self.labelFiles)
            ]
            results = pool.starmap(load_data, args)

        for batch_data, batch_label in results:
            batch_data_list.append(batch_data.reshape(500, -1))
            batch_label_list.append(batch_label)

        batch_data_list = np.array(batch_data_list).reshape(-1, 500)
        batch_label_list = np.array(batch_label_list).reshape(-1, 1)

        if self.end_index == self.length:
            self.start_index = 0
            self.end_index = 0


        return batch_data_list, batch_label_list
    
    def __getitem__(self, index):
        # Check if current bin is exhausted
        if self.count >= self.bin_size or self.count < 0:
            self.count = 0
            self.data, self.labels = self.loadbin()
        self.count += 1
        self.count %= self.bin_size
        
        # Get data for the current index
        data = self.data[self.count]
        label = self.labels[self.count]
        
        return data, label
    
def load_data(data_file, label_file, start_index, bin_size):
    chrData = []
    for chrFile in data_file:
        data = pd.read_csv(
            chrFile[1],
            delimiter=" ",
            header=None,
            skiprows=start_index,
            nrows=bin_size,
        )
        data = data.values.astype(np.float32)
        data = np.multiply(data.T, int(chrFile[0]))
        chrData.append(data)

    chrData = np.array(chrData)

    # get the values
    label = pd.read_csv(
        label_file,
        delimiter=" ",
        header=None,
        skiprows=start_index,
        nrows=bin_size,
    )
    label = label.values.astype(np.float32)

    return chrData, label

def getData(
    trainLabel="chr11-chr7",
    testLabel="chr11",
    validLabel="chr7",
    fileLocation="./Data/220802_DATA/",
    chrDrop=None,
    cellLineDrop=None,
    bin_size=256,
    dataTypes="-1",
):
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    chr_train = []
    chr_test = []
    chr_valid = []

    # Create the datasets
    train = Chromatin_Dataset(
        cellLineDrop,
        chrDrop,
        trainLabel,
        fileLocation + "/TRAIN/",
        "train",
        bin_size // len(cellLineDrop),
        dataTypes,
    )
    test = Chromatin_Dataset(
        cellLineDrop,
        chrDrop,
        testLabel,
        fileLocation + "/HOLDOUT/",
        "test",
        bin_size // len(cellLineDrop),
        dataTypes,
    )
    valid = Chromatin_Dataset(
        cellLineDrop,
        chrDrop,
        validLabel,
        fileLocation + "/HOLDOUT/",
        "valid",
        bin_size // len(cellLineDrop),
        dataTypes,
    )
    chr_train.append(train)
    chr_test.append(test)
    chr_valid.append(valid)

    # Concatenate the datasets
    train = torch.utils.data.ConcatDataset(chr_train)
    test = torch.utils.data.ConcatDataset(chr_test)
    valid = torch.utils.data.ConcatDataset(chr_valid)

    return train, test, valid
