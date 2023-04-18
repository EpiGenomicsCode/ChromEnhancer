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
        super(Chromatin_Dataset, self).__init__()

        self.cellLines = ["A549", "MCF7", "HepG2", "K562"]
        self.chromatine = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

        # Saving the parameters
        self.cellLinesDrop = cellLinesDrop
        self.chrDrop = chromTypesDrop
        self.label = label
        self.file_location = file_location
        self.dataUse = dataUse
        self.dataTypes = dataTypes

        # initialize data and label variables
        self.data = None
        self.labels = None

        # Load in file paths for data and label files
        self.dataFiles = np.array(self.getDataFiles())
        self.labelFiles = np.array(self.getLabelFile())

        self.length = self.getNumSamples()


        # initialize bining variables
        self.count = -1
        self.countSize = 0
        self.start_index = 0
        self.bin_size = max(bin_size//(len(self.cellLines)-len(self.cellLinesDrop)), 1)
        self.end_index =0

        self.reset = False
        # print all datafiles as an indented string
        print(f"Usage: {self.dataUse}")
        print(f"length: {self.length}")
        print(f"bin_size: {self.bin_size} per cellLine")
        for index in range(len(self.labelFiles)):
            print(f"\tLabel: {self.labelFiles[index]}")
            files = self.dataFiles[index]
            for file in files:
                print(f"\t\tData: {file[1]}\n\t\t\tkeep: {file[0]}")
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
        #  remove duplicates from datafiles
        datafiles = list(datafiles for datafiles, _ in itertools.groupby(datafiles))

        return datafiles

    def loadDataFiles(self):
        batch_data_list = []
        for cellLine in self.dataFiles:
            cellLineData = []

            # CTCF-1, H3K4me3-1, H3K27ac-1, p300-1, PolII-1
            for chrFile in cellLine:
                data = pd.read_csv(chrFile[1], sep=" ", header=None, skiprows=self.start_index,nrows=self.bin_size ).values.astype(np.float32)
                data = np.multiply(data, int(chrFile[0])) #  multiply by 0 or 1 to remove data
                cellLineData.append(data)

            
            cellLineData = np.concatenate(np.array(cellLineData), axis=1)
            batch_data_list.append(cellLineData)
        batch_data_list = np.concatenate(np.array(batch_data_list))
        
        return batch_data_list
    
    
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

    def loadLabelFiles(self):
        batch_label_list = []
        for labelFile in self.labelFiles:
            batch_label_list.append(pd.read_csv(labelFile, sep=" ", header=None, skiprows=self.start_index,nrows=self.bin_size ).values)
        batch_label_list = np.concatenate(np.array(batch_label_list))
        return batch_label_list

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
        batch_data_list = self.loadDataFiles()
        batch_label_list = self.loadLabelFiles()

        self.countSize = batch_data_list.shape[0]

        return batch_data_list, batch_label_list
        
    def __getitem__(self, index):

        # Increment the count
        self.count += 1

        # Check if we have reached the end of the dataset and reset the indices
        if self.count >= self.length:
            self.end_index = 0
            self.start_index = 0
            self.count = -1
            self.countSize = 0
            self.reset = True

        # Check if we need to load the next bin
        if self.count == self.countSize or self.count == -1:
            self.data, self.labels = self.loadbin()

        # Get the current data and labels for the current index
        data = self.data[index % self.bin_size]
        label = self.labels[index % self.bin_size]

        return np.round(data,5), label
    

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
        bin_size,
        dataTypes,
    )
    test = Chromatin_Dataset(
        cellLineDrop,
        chrDrop,
        testLabel,
        fileLocation + "/HOLDOUT/",
        "test",
        bin_size,
        dataTypes,
    )
    valid = Chromatin_Dataset(
        cellLineDrop,
        chrDrop,
        validLabel,
        fileLocation + "/HOLDOUT/",
        "valid",
        bin_size,
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
