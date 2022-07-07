from pydoc import describe
from typing import Dict
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import itertools
import pandas as pd
import pdb
class Chromatin_Dataset(Dataset):
    def __init__(self, chromType="chr10-chr17", chromName="CTCF-1", file_location="./Data/220627_DATA/TRAIN/*"):
        super(Dataset, self).__init__()
        self.data, self.labels = self.readfiles(chromType, chromName, file_location)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index],dtype=torch.float32)

    def make_onehot(self, sequences, seq_length=32):    
        fd = {  'A' : [1, 0, 0, 0], 'T' : [0, 1, 0, 0], 'G' : [0, 0, 1, 0], 'C': [0, 0, 0, 1], 
                'N' : [0, 0, 0, 0], 'a' : [1, 0, 0, 0], 't' : [0, 1, 0, 0],
                'g' : [0, 0, 1, 0], 'c' : [0, 0, 0, 1],
                'n' : [0, 0, 0, 0]
             }
        onehot = [fd[base] for seq in sequences for base in seq]
        onehot_np = np.reshape(onehot, (-1, seq_length, 4))
        return onehot_np


    def readfiles(self,chromeType,chromName, file_location):
        files = glob(file_location)
        labels = []
        data = []

        for i in files:
            if chromeType in i and chromName in i:
                if ".chromtrack" in i:
                    print("Processing: {}".format(i))
                    value = []
                    with open(i) as openfile:
                        for line in openfile:
                            data.append([float(ii) for ii in line.strip().split(" ")])
                    data = np.array(data)
            if chromeType in i:
                if ".label" in i:
                    print("Processing: {}".format(i))
                    value = []
                    with open(i) as openfile:
                        for line in openfile:
                            value.append(float(line.strip()))
                    labels = np.array(value)

        
        return data, labels