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

    def __init__(
            self,
            id,
            chromTypes,
            label,
            file_location, 
            dataUse , 
            drop=None):
        """
        input:
        =====
            id: string: Chromatin Cell identification
            chromType: list of strings: Order of chromatin types
            label: string: Enhancer labels
            file_location: string
            dataUse: string: train, test or valid dataset
            drop: string: Chromatin type to drop
        """
        super(Dataset, self).__init__()

        # self.length = int(subprocess.check_output("wc -l {}".format(self.labelFilenames), shell=True).decode().split()[0])
        # load in every file for chromType
        self.data = []
        self.dataFiles = []
        self.label = []
        self.labelFiles = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for chromType in chromTypes:
            if chromType != drop:
                fileFormat = file_location + "{}*_*{}*_{}*.chromtrack".format(id, label, chromType)
                # Load in data
                file = glob(fileFormat)[0]
                data = pd.read_csv(file, delimiter=" ", header=None).values.astype(np.float32)
                self.dataFiles.append(file)
                # Transpose data
                data = data.T
                # Add data to self.data
                self.data.append(data)
        
        # self.data is shape of 5 x 100 x 641883 which is 5 chromatin types, 100 features, 641883 samples
        self.data = np.array(self.data)
        
        # Load in label data
        labelName = glob(file_location + "{}*_*{}*.label".format(id, label))[0]
        self.labelFiles.append(labelName)
        self.label = pd.read_csv(labelName, delimiter=" ", header=None)

        # convert to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32).to(device)
        self.label = torch.tensor(np.array(self.label.values), dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[:, :, index].flatten(),self.label[index]
        

def getData(chromtypes     = [
                                "CTCF-1",
                                "H3K4me3-1",
                                "H3K27ac-1",
                                "PolII-1"
                            ], 
            id            = "A549", 
            trainLabel    = "chr11-chr7", 
            testLabel     = "chr11", 
            validLabel    = "chr7",
            fileLocation  = "./Data/220802_DATA/",
            drop=None
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
    # Load the training data
    chr_train = Chromatin_Dataset(
        id=id,
        chromTypes=chromtypes,
        label=trainLabel,
        file_location=fileLocation+"/TRAIN/*", dataUse="train", drop=drop)

    # Load the test data
    chr_test = Chromatin_Dataset(
        id=id,
        chromTypes=chromtypes,
        label=testLabel,
        file_location=fileLocation+"/HOLDOUT/*", dataUse="test", drop=drop)

    # Load the validation data
    chr_valid = Chromatin_Dataset(
            id=id,
            chromTypes=chromtypes,
            label=validLabel,
            file_location=fileLocation+"/HOLDOUT/*", dataUse="valid", drop=drop)
    
    # Create output directory
    os.makedirs('./output', exist_ok=True)
    

    return chr_train, chr_test, chr_valid   