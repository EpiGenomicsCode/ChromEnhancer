from pydoc import describe
from typing import Dict
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import itertools
import pandas as pd
from Model.util import readfiles

class Chromatin_Dataset(Dataset):
    def __init__(self, chromType="chr10-chr17", chromName="CTCF-1", file_location="./Data/220627_DATA/TRAIN/*"):
        """
        chromType = label
        chronName = chrome
        """
        super(Dataset, self).__init__()
        self.data, self.labels = readfiles(chromType, chromName, file_location)
        self.filename = chromType+"_"+chromName
    

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index],dtype=torch.float32)
