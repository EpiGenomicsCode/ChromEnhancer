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
from multiprocessing import Pool

class Chromatin_Dataset(Dataset):
    def __init__(
        self,
        cellLinesDrop,
        chromTypesDrop,
        label,
        file_location,
        dataUse,
        bin_size=10,
        dataTypes="-1",
    ):
        super(Chromatin_Dataset, self).__init__()

        self.cellLines = ["A549", "MCF7", "HepG2", "K562"]
        self.chromatine = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

        # load in every file for chromType
        self.data = []
        self.dataFiles = []
        self.label = []
        self.labelFiles = []
        id = [i for i in self.cellLines if i not in cellLinesDrop][0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for chromType in [i for i in self.chromatine if i not in chromTypesDrop]:
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
        labelNames = glob(file_location + "{}*_*{}*.label".format(id, label))
        if dataUse == "train":
              labelName = [
                        i for i in labelNames if  id in i and label in i and not "Leniant" in i and not "Stringent" in i 
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
        self.label = pd.read_csv(self.labelFile, delimiter=" ", header=None)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return np.array(self.data[:, :, index]).flatten(), np.array(self.label.iloc[index])

