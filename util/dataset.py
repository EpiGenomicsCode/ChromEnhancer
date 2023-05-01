import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class Chromatin_Dataset(Dataset):
    def __init__(
        self,
        cellLine,
        chrUse,
        dataTypes,
        label,
        fileLocation,
        chunk_size=4096,
        mode="train",
    ):
        """
        Args:
            cellLine (str): cell line name.
            chrUse (list): list of chromatin types to use.
            dataTypes (str): data type to use.
            label (str): label name.
            fileLocation (str): path to the data files.
            mode (str): dataset mode (train, test or valid).
            chunk_size (int): chunk size to use when loading the HDF5 data. Default: 1000.
        """
        super(Chromatin_Dataset, self).__init__()

        self.cellLine = cellLine
        self.chrUse = chrUse
        self.dataTypes = dataTypes
        self.label = label
        self.fileLocation = fileLocation
        self.mode = mode
        self.chromatin =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
        self.missing = [i for i in range(len(self.chromatin)) if self.chromatin[i] not in self.chrUse]
        self.chunk_size = chunk_size
        self.start_chunk = 0
        self.end_chunk = self.chunk_size
        self.chunk_counter = 0
        if "220802" in fileLocation:
            if self.mode == "train":
                self.dataName = f"{self.cellLine}_{self.label}_{self.dataTypes}"
                self.labelName = f"{self.cellLine}_{self.label}_labels"
            elif self.mode == "test":
                self.dataName = f"{self.cellLine}_{self.label}_{self.dataTypes}"
                self.labelName = f"{self.cellLine}_{self.label}_StringentEnhancer_labels"
            else:
                self.dataName = f"{self.cellLine}_{self.label}_{self.dataTypes}"
                self.labelName = f"{self.cellLine}_{self.label}_LenientEnhancer_labels"
        elif "220803" in fileLocation:
            if self.mode == "train":
                self.dataName = f"{self.cellLine}_{self.dataTypes}"
                self.labelName = f"{self.cellLine}_labels"
            elif self.mode == "test":
                self.dataName = f"{self.cellLine}_train"
                self.labelName = f"{self.cellLine}_StringentEnhancer"
            else:
                self.dataName = f"{self.cellLine}_{self.label}_{self.dataTypes}"
                self.labelName = f"{self.cellLine}_LenientEnhancer"
        elif "Sequence" in fileLocation:
            if self.mode == "train":
                self.dataName = f"{self.cellLine}_{self.label}_train"
                self.labelName = f"{self.cellLine}_{self.label}_train"
            elif self.mode == "test":
                self.dataName = f"{self.cellLine}_StringentEnhancer_{self.label}"
                self.labelName = f"{self.cellLine}_StringentEnhancer_{self.label}"
            else:
                self.dataName = f"{self.cellLine}_LenientEnhancer_{self.label}"
                self.labelName = f"{self.cellLine}_LenientEnhancer_{self.label}"
        elif "230124" in fileLocation:
            if self.mode == "train":
                self.dataName = f"{self.cellLine}_{self.label}_train"
                self.labelName = f"{self.cellLine}_train"
            elif self.mode == "test":
                self.dataName = f"{self.cellLine}_StringentEnhancer_{self.label}"
                self.labelName = f"{self.cellLine}_StringentEnhancer"
            else:
                self.dataName = f"{self.cellLine}_LenientEnhancer_{self.label}"
                self.labelName = f"{self.cellLine}_LenientEnhancer"


        self.DataFile, self.LabelFile = self.getFiles()
        self.Dataset = h5py.File(self.DataFile, 'r')[self.dataName]
        self.Labelset = h5py.File(self.LabelFile, 'r')[self.labelName]
        assert len(self.Dataset) == len(self.Labelset), "Data and label lengths do not match."
        self.length = len(self.Dataset)


    def getFiles(self):
        """
        Returns the file names for the data and labels.
        """
        DataFile = glob(f"{self.fileLocation}*D*.h5")[0]
        LabelFile = glob(f"{self.fileLocation}*L*.h5")[0]
        return DataFile, LabelFile

    def __len__(self):
        return self.length

    def convertSequence(self, sequence, alphabet='ACGT'):
        return torch.tensor([[1 if letter == alphabet[i] else 0 for i in range(len(alphabet))] for letter in sequence])

    def __getitem__(self, index):
        """
        Returns a data-label pair.
        """
        # find the index of the missing chromatin type
        missing_index = [i for i in range(len(self.chromatin)) if self.chromatin[i] not in self.chrUse]

        if self.chunk_counter % self.chunk_size == 0:
            self.start_chunk = self.end_chunk
            self.end_chunk = min(self.end_chunk + self.chunk_size, self.length)
            # load the next chunk
            self.Dataset = h5py.File(self.DataFile, 'r')[self.dataName][self.start_chunk:self.end_chunk]
            self.Labelset = h5py.File(self.LabelFile, 'r')[self.labelName][self.start_chunk:self.end_chunk]


        if self.chunk_counter >= self.length or self.start_chunk >= self.end_chunk:
            self.chunk_counter = 0
            self.start_chunk = 0
            self.end_chunk = self.chunk_size
            self.Dataset = h5py.File(self.DataFile, 'r')[self.dataName][self.start_chunk:self.end_chunk]
            self.Labelset = h5py.File(self.LabelFile, 'r')[self.labelName][self.start_chunk:self.end_chunk]


        pos = index % (self.end_chunk - self.start_chunk)
        
        data = self.Dataset[pos]
        label = self.Labelset[pos]

        if self.mode == "train":
            # zero out the missing chromatin type
            for i in missing_index:
                data[i*100:(i+1)*100] = 0

        self.chunk_counter += 1

        if "Sequence" in self.fileLocation:
            return self.convertSequence(data[0]).flatten(), label
        else:
            return data, label


        

def getData(
    trainLabel="chr11-chr7",
    testLabel="chr11",
    validLabel="chr7",
    dataTypes="-1",
    fileLocation="./Data/220802_DATA/",
    chrUse=None,
    cellLineUse=None,
    bin_size=4096
):
    train = []
    test = []
    valid = []

    for cellLine in cellLineUse:
        train.append(
            Chromatin_Dataset(
                cellLine,     # A549
                chrUse,       # ["CTCF", "H3K27ac", "H3K27me3"...]
                dataTypes,    # -1
                trainLabel,   # chr11-chr7
                fileLocation+"TRAIN/", # ./Data/220802_DATA/
                bin_size,
                "train")
        )
        test.append(
            Chromatin_Dataset(
                cellLine,     # A549
                chrUse,       # ["CTCF", "H3K27ac", "H3K27me3"...]
                dataTypes,    # -1
                testLabel,   # chr11-chr7
                fileLocation+"HOLDOUT/", # ./Data/220802_DATA/
                bin_size,
                "test")
        )

        valid.append(
            Chromatin_Dataset(
                cellLine,     # A549
                chrUse,       # ["CTCF", "H3K27ac", "H3K27me3"...]
                dataTypes,    # -1
                validLabel,   # chr11-chr7
                fileLocation+"HOLDOUT/", # ./Data/220802_DATA/
                bin_size,
                "valid")
               
        )
    train = torch.utils.data.ConcatDataset(train)
    test = torch.utils.data.ConcatDataset(test)
    valid = torch.utils.data.ConcatDataset(valid)

    return train, test, valid