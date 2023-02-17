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


class Sequence_Dataset(Dataset):
    """
    Dataset class for the Chromatin data
    """

    def __init__(self, 
                filename, type="train"):
        """
        input:
        =====
           filename: the filename of the data
        """
        
        super(Dataset, self).__init__()
        # process Filename
        self.type = type
        if self.type == "train":
            self.filename = filename
            filename = filename[filename.rindex("/")+1:filename.rindex(".")]
            self.labelName = "./Data/220802_DATA/TRAIN/"+filename+".label"


        elif self.type == "test":
            id = filename.split("/")[-1].split("_")[0]
            test_chr = filename[filename.index("chr"):filename.rindex("chr")-1].strip()
            self.filename = "./Data/230124_CHR-Data_Sequence/CHR-CHROM/HOLDOUT/{}_StringentEnhancer_{}.seq".format(id, test_chr)
            self.labelName = "./Data/220802_DATA/HOLDOUT/{}_StringentEnhancer_{}.label".format(id, test_chr)
            
        else:
            id = filename.split("/")[-1].split("_")[0]
            test_chr = filename[filename.index("chr"):filename.rindex("chr")-1].strip()
            self.filename = "./Data/230124_CHR-Data_Sequence/CHR-CHROM/HOLDOUT/{}_LenientEnhancer_{}.seq".format(id, test_chr)
            self.labelName = "./Data/220802_DATA/HOLDOUT/{}_LenientEnhancer_{}.label".format(id, test_chr)

        print("\n\n\n")
        print("filename: ", self.filename)
        print("labelName: ", self.labelName)

        # Read in the data
        self.data = []
        # read in the file line by line and add it to data
        with open(self.filename, "r") as f:
            for line in f:
                seq = line.strip()
                self.data.append(seq)

        # Read in the label
        self.label = pd.read_csv(self.labelName, delimiter=" ", header=None)
        

        
    def __len__(self):
        return min(len(self.label), 2**10)

    def __getitem__(self, index):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return one_hot_encode(self.data[index]).to(device).flatten().to(torch.float32), torch.tensor(self.label.iloc[index]).to(device).to(torch.float32)

def one_hot_encode(sequence, alphabet='ACGT'):
    encoded = torch.tensor([[1 if letter == alphabet[i] else 0 for i in range(len(alphabet))] for letter in sequence])
    return encoded