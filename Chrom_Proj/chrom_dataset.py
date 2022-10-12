from pydoc import describe
from typing import Dict
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import itertools
import pandas as pd
from Chrom_Proj.util import readfiles


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
        self.data, self.labels = readfiles(id, chromType, label, file_location)
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
