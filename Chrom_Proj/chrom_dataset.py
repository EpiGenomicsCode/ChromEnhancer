from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import itertools
import pandas as pd
import pdb

class Chromatin_Dataset(Dataset):
    """
    This Chromatine Dataset for pytorch
    """
    def __init__(
            self,
            id="A549",
            chromType=[
                "CTCF-1",
                "H3K4me3-1",
                "H3K27ac-1",
                "p300-1",
                "PolII-1"],
            label="chr10-chr17",
            file_location="./Data/220802_DATA/TRAIN/*"):
        """
        initalizer function:
            Input:
                id: String: the Chromatine Name
                chromType: List of Strings: the order for chromatin
                label: String: the training data to use
                file_location: String: Location of the dataset
        """
        super(Dataset, self).__init__()
        self.data, self.labels = readFiles(id, chromType, label, file_location)
        self.data = self.data
        self.labels = self.labels
        self.filename = id + "_" + label

        assert len(self.data) == len(self.labels)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(
            self.data[index], dtype=torch.float32), torch.tensor(
            self.labels[index], dtype=torch.float32)

def readFiles(id, chromType, label, file_location):
    """
        Reads preprocessed files and returns the data and labels

        input:
        =====
            id: string
            chromType: list of strings
            label: string: Enhancer labels
            file_location: string
    """
    print(file_location)
    files = glob(file_location)
    labels = []
    data = {}

    for fileType in chromType:
        filename = [
            i for i in files if id in i and fileType in i and "chromtrack" in i and label in i]
        assert len(filename) != 0
        print("Processing: {}".format(filename[0]))
        fileName = filename[0]

        data[fileType] = pd.read_csv(fileName, delimiter=" ",  header=None)

    horizontalConcat = pd.DataFrame()
    for fileType in chromType:
        horizontalConcat = pd.concat(
            [horizontalConcat, data[fileType]], axis=1)

    labelFileName = [
                        i for i in files if ".label" in i and id in i and label in i
                    ]
    assert len(labelFileName) > 0
    print("Processing: {}".format(labelFileName[0]))
    label = pd.read_csv(labelFileName[0], delimiter=" ", header=None)

    return np.array(horizontalConcat.values[:]), np.array(label.values[:])

def getData(chromtypes, 
            id, 
            trainLabel, 
            testLabel, 
            validLabel,
            fileLocation="./Data/220802_DATA"
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
