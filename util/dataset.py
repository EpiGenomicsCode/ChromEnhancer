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


class Chromatin_Dataset(Dataset):
    def __init__(self,
                 cellLinesDrop,
                 chromTypesDrop,
                 label,
                 file_location,
                 dataUse, 
                bin_size=10, 
                dataTypes ="-1"):
        super().__init__()

        
        self.cellLines = ["A549", "MCF7", "HepG2", "K562"]
        self.chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

        # Saving the parameters
        self.cellLinesDrop= cellLinesDrop
        self.chrDrop = chromTypesDrop
        self.label = label
        self.file_location = file_location
        self.dataUse = dataUse
        self.dataTypes = dataTypes

        # initialize bining variables
        self.count = 0
        self.start_index = 0
        self.bin_size = bin_size
        self.end_index = self.bin_size

        # initialize data and label variables
        self.data = None
        self.labels = None
        self.labelFile = None
        self.num_samples = None


        # Load in file paths for data and label files
        self.dataFiles = np.array(self.getDataFiles())
        self.labelFiles = np.array(self.getLabelFile())

        # print all datafiles as an indented string
        for datafile in self.dataFiles:
            for labelfile in self.labelFiles:
                print(labelfile)
                for chr in datafile:
                    print("\t", chr[1])
        print(f"Usage: {self.dataUse}")
        print("=================================\n")

                

        # get the number of samples
        self.num_samples = self.getNumSamples()


    def getDataFiles(self):
        datafiles = []

        for cellLine in self.cellLines:
            cellLineFiles = []
            for chr in self.chromatine:
                if cellLine not in self.cellLinesDrop:
                    files = glob(f"{self.file_location}/*{cellLine}*{self.label}*{chr}*{self.dataTypes}*")
                    try:
                        if chr not in self.chrDrop:
                            cellLineFiles.append([1, files[0]])
                        else:
                            cellLineFiles.append([0, files[0]])
                    except:
                        pdb.set_trace()
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
                    labelName = [f for f in files if "Leniant" not in f and "Stringent" not in f][0]
                if self.dataUse == "test":
                    labelName = [f for f in files if "Lenient" in f][0]
                if self.dataUse == "valid":
                    labelName = [f for f in files if "Stringent" in f][0]
                if len(labelName) != 0:
                    labelNames.append(labelName)
                
        return labelNames
    
    def getNumSamples(self):
        numSamples = []
        for labelFile in self.labelFiles:
            numSamples.append(sum(1 for line in open(labelFile)))
        return sum(numSamples)

    def __len__(self):
        return self.num_samples // self.bin_size

    def __getitem__(self, index):
        # Calculate the start and end indices for this bin
        self.start_index = index * self.bin_size
        self.end_index = min(self.start_index + self.bin_size, self.num_samples)

        # Load data for this bin
        batch_data_list = []
        batch_label_list = []
        
        for data_file, label_file in zip(self.dataFiles, self.labelFiles):
            chrData = []
            for chrFile in data_file:
                data = pd.read_csv(chrFile[1], delimiter=" ", header=None, skiprows=self.start_index, nrows=self.bin_size)
                data = data.values.astype(np.float32)
                data = np.multiply(data.T, int(chrFile[0]))         
                chrData.append(data)
            chrData = np.array(chrData).reshape(self.bin_size, -1)
            batch_data_list.append(chrData)
            
            # get the values
            label = pd.read_csv(label_file, delimiter=" ", header=None, skiprows=self.start_index, nrows=self.bin_size)
            label = label.values.astype(np.float32)

            # save the data
            batch_label_list.append(label)

        batch_data = np.concatenate(np.array(batch_data_list))
        label = np.concatenate(np.array(batch_label_list))

        return batch_data, label


def getData(
            trainLabel    = "chr11-chr7", 
            testLabel     = "chr11", 
            validLabel    = "chr7",
            fileLocation  = "./Data/220802_DATA/",
            chrDrop=None, 
            cellLineDrop=None,
            bin_size=256, 
            dataTypes = "-1"
        ):
    

    # Create output directory
    os.makedirs('./output', exist_ok=True)
    chr_train = []
    chr_test = []
    chr_valid = []
   
    # Create the datasets
    train = Chromatin_Dataset(cellLineDrop, chrDrop, trainLabel, fileLocation+"/TRAIN/", "train",  bin_size, dataTypes)
    test = Chromatin_Dataset(cellLineDrop, chrDrop, testLabel, fileLocation+"/HOLDOUT/", "test",  bin_size, dataTypes)
    valid = Chromatin_Dataset(cellLineDrop, chrDrop, validLabel, fileLocation+"/HOLDOUT/", "valid",  bin_size,  dataTypes)

    chr_train.append(train)
    chr_test.append(test)
    chr_valid.append(valid)

    # Concatenate the datasets
    train = torch.utils.data.ConcatDataset(chr_train)
    test = torch.utils.data.ConcatDataset(chr_test)
    valid = torch.utils.data.ConcatDataset(chr_valid)

    return train, test, valid

