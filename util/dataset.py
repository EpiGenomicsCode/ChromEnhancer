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


class Chromatin_Dataset(Dataset):
    """
    Dataset class for the Chromatin data
    """

    def __init__(self, id, chromTypes, label, file_location, dataUse, drop=None, batch_size=1000):
        super(Chromatin_Dataset, self).__init__()
        
        self.id = id
        self.chromTypes = chromTypes
        self.label = label
        self.file_location = file_location
        self.dataUse = dataUse
        self.drop = drop
        self.batch_size = batch_size
        
        # Load in file paths for data and label files
        self.dataFiles = []
        self.labelFiles = []
        for chromType in chromTypes:
            fileFormat = file_location + "{}*_*{}*_{}*.chromtrack".format(id, label, chromType)
            file = glob(fileFormat)[0]
            if chromType == self.drop and self.dataUse == "train":
                self.dataFiles.append(f"None_{file}")
            else:
                self.dataFiles.append(file)
        
        labelNames = glob(file_location + "{}*_*{}*.label".format(id, label))
        if dataUse == "train":
            labelName = [
                i for i in labelNames if  id in i and label in i and not "Lenient" in i and not "Stringent" in i 
            ][0]
        elif dataUse == "test":
            labelName = [
                i for i in labelNames if ".label" in i and id in i and label in i and not "Lenient" in i 
            ][0]
        else:
            labelName = [
                i for i in labelNames if ".label" in i and id in i and label in i and not "Stringent" in i 
            ][0]
        self.labelFile = labelName

        # Compute length of dataset using system call to wc
        with open(self.labelFile, 'r') as f:
            self.num_samples = len(f.readlines())
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Load data for this batch
        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, self.num_samples)
        batch_data = []
        for data_file in self.dataFiles:
            if "None" in data_file:
                filename = data_file[data_file.index("None"):]
                data = pd.read_csv(data_file, delimiter=" ", header=None, skiprows=index-self.batch_size, nrows=self.batch_size).values.astype(np.float32)
                data = data.T
                data = np.array([np.zeros(data.shape[1])])
            else:
                data = pd.read_csv(data_file, delimiter=" ", header=None, skiprows=index-self.batch_size, nrows=self.batch_size).values.astype(np.float32)
                data = data.T

            batch_data.append(data)

        batch_data = np.array(batch_data)
        # shuffle the rows
        batch_data = batch_data[:, np.random.permutation(batch_data.shape[1]), :]

        # Load labels for this batch
        label = pd.read_csv(self.labelFile, delimiter=" ", header=None, skiprows=index-self.batch_size, nrows=self.batch_size)

        return batch_data.reshape(-1), np.array(label)


def getData(chromtypes     = [
                                "CTCF-1",
                                "H3K4me3-1",
                                "H3K27ac-1",
                                "PolII-1"
                            ], 
            ids            = ["A549"], 
            trainLabel    = "chr11-chr7", 
            testLabel     = "chr11", 
            validLabel    = "chr7",
            fileLocation  = "./Data/220802_DATA/",
            drop=None, 
            batch_size=2048
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
            (ex: ./Data/220802_DATA/)

    Return:
        trainer: training data
        
        tester: testing data
        
        validator: validation data
    """

    # Create output directory
    os.makedirs('./output', exist_ok=True)
    chr_train = []
    chr_test = []
    chr_valid = []
    print("Generating data for: ", ids)

    for id in ids:
        # Load the training data
        chr_train.append(Chromatin_Dataset(
            id=id,
            chromTypes=chromtypes,
            label=trainLabel,
            file_location=fileLocation+"/TRAIN/*", dataUse="train", drop=drop, batch_size=batch_size))

        # Load the test data
        chr_test.append(Chromatin_Dataset(
            id=id,
            chromTypes=chromtypes,
            label=testLabel,
            file_location=fileLocation+"/HOLDOUT/*", dataUse="test", drop=drop, batch_size=batch_size))

        # Load the validation data
        chr_valid.append(Chromatin_Dataset(
                id=id,
                chromTypes=chromtypes,
                label=validLabel,
                file_location=fileLocation+"/HOLDOUT/*", dataUse="valid", drop=drop, batch_size=batch_size))
    # if we are doing the celline dropout we need to include all the data
    if len(ids) != 1:
        all_data = ["A549", "MCF7", "HepG2", "K562"]
        # find the id that is not in the list
        newids = [i for i in all_data if i not in ids]
        print("Validation data includes {}".format(newids))
        for id in newids:
            # Load the extra validation data
            chr_valid.append(Chromatin_Dataset(
                    id=newids[0],
                    chromTypes=chromtypes,
                    label=validLabel,
                    file_location=fileLocation+"/HOLDOUT/*", dataUse="valid", drop=drop))
        
    
    chr_train = torch.utils.data.ConcatDataset(chr_train)
    chr_test = torch.utils.data.ConcatDataset(chr_test)
    chr_valid = torch.utils.data.ConcatDataset(chr_valid)

    return chr_train, chr_test, chr_valid   