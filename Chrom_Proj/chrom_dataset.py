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
            file_location="./Data/220802_DATA/TRAIN/*", dataUse="train", drop=None):
        """
        initalizer function:
            Input:
                id: String: the Chromatine Name
                chromType: List of Strings: the order for chromatin
                label: String: the training data to use
                file_location: String: Location of the dataset
        """
        super(Dataset, self).__init__()
        self.dataFilenames, self.labelFilenames = readFiles(id, chromType, label, file_location, dataUse)
        self.length = int(subprocess.check_output("wc -l {}".format(self.labelFilenames), shell=True).decode().split()[0])
        print("\t length: {}".format(self.length))
        self.dataIterator = {}
        self.data = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drop = drop
        self.loadChunk()
        self.nextChunk()

    def loadChunk(self):
        for key in self.dataFilenames.keys():
            self.dataIterator[key] = pd.read_csv(self.dataFilenames[key], delimiter=" ", header=None, chunksize=self.length//1000)
        self.dataIterator["label"] = pd.read_csv(self.labelFilenames, delimiter=" ", header=None, chunksize=self.length//1000)

    def nextChunk(self):
        for key in self.dataIterator.keys():
            if key == self.drop:
                self.data[key][:] = 0
            
            self.data[key] = torch.tensor(next(self.dataIterator[key]).values,dtype=torch.float32).to('cuda')
            
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cellLine = []
        if len(self.data[list(self.data.keys())[0]]) == 0:
            self.nextChunk()
                
        for key in [i for i in self.data.keys() if 'label' not in i]:
            cellLine.append(self.data[key][0])
            self.data[key] = self.data[key][1:,:]

        label = self.data['label'][0]
        self.data['label'] = self.data['label'][1:,:]

        
        return torch.cat(cellLine), label

        

def readFiles(id, chromType, label, file_location, dataUse):
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
        # print("Processing: {}".format(filename[0]))
        fileName = filename[0]
        data[fileType] = fileName

    if dataUse == "train":
        labelFileName = [
                        i for i in files if ".label" in i and id in i and label in i and not "Leniant" in i and not "Stringent" in i 
                    ]
    if dataUse == "test":
        labelFileName = [
                        i for i in files if ".label" in i and id in i and label in i and not "Lenient" in i 
                    ]
    if dataUse == "valid":
        labelFileName = [
                        i for i in files if ".label" in i and id in i and label in i and not "Stringent" in i 
                    ]
    

    print("using label :{}".format(labelFileName[0]))

    return data, labelFileName[0]

def getData(chromtypes, 
            id, 
            trainLabel, 
            testLabel, 
            validLabel,
            fileLocation="./Data/220802_DATA",
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
            (ex: ./Data/220708/DATA)

    Return:
        trainer: training data
        
        tester: testing data
        
        validator: validation data
    """
    os.makedirs('./output', exist_ok=True)
    
    chr_train = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=trainLabel,
        file_location=fileLocation+"/TRAIN/*", dataUse="train", drop=drop)

    chr_test = Chromatin_Dataset(
        id=id,
        chromType=chromtypes,
        label=testLabel,
        file_location=fileLocation+"/HOLDOUT/*", dataUse="test", drop=drop)

    chr_valid = Chromatin_Dataset(
            id=id,
            chromType=chromtypes,
            label=validLabel,
            file_location=fileLocation+"/HOLDOUT/*", dataUse="valid", drop=drop)
   
    return chr_train, chr_test, chr_valid   
