from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch

class Chromatin_Dataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.data, self.labels = self.readfiles()
        self.input_shape = len(self.data[0])
        self.data = self.data
        self.labels = self.labels

        assert len(self.data) == len(self.labels)

        self.length = len(self.data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def make_onehot(self, sequences, seq_length=32):    
        fd = {  'A' : [1, 0, 0, 0], 'T' : [0, 1, 0, 0], 'G' : [0, 0, 1, 0], 'C': [0, 0, 0, 1], 
                'N' : [0, 0, 0, 0], 'a' : [1, 0, 0, 0], 't' : [0, 1, 0, 0],
                'g' : [0, 0, 1, 0], 'c' : [0, 0, 0, 1],
                'n' : [0, 0, 0, 0]
             }
        onehot = [fd[base] for seq in sequences for base in seq]
        onehot_np = np.reshape(onehot, (-1, seq_length, 4))
        return onehot_np


    def removeDuplicated(self, data, label):
        dataset = set()
        newData =[]
        newLabel =[]
        for data in zip(data, label):
            molData = data[0]
            molLab  = data[1]
            if not molData in dataset:
                newData.append(molData)
                newLabel.append(molLab)
                dataset.add(molData)
            else:
                print("duplicate:{}\t{}".format(molLab, newData))

        return newData, newLabel

    def readfiles(self, file_location="../Data/Bichrom_sample_data/"):
        chrom_filenames = glob(file_location+"*chrom*")
        label_filenames = glob(file_location+"*label*")
        seq_filenames   = glob(file_location+"*seq*" )

        data  = []
        label = []
        seq   = []
        print("Processing\n\tchrome files {}\n\tlabel files {}".format(chrom_filenames, label_filenames))
        for files in zip(chrom_filenames, label_filenames, seq_filenames):
            chrom_file = files[0]
            label_file = files[1]
            seq_file   = files[2]
            if "val" not in chrom_file:
                chrom_len = 0
                label_len = 0
                seq_len   = 0
                with open(chrom_file) as infile:
                    for line in infile:
                        chrom_len+=1
                        data.append(torch.tensor(np.array([float(i) for i in line.split()]),dtype=torch.float32))
                with open(label_file)  as infile:
                    for line in infile:
                        label_len+=1
                        label.append(torch.tensor(np.array(int(line)), dtype=torch.float32))
                # with open(seq_file)  as infile:
                #     for line in infile:
                #         seq_len+=1
                #         seq.append(torch.tensor(self.make_onehot(line), dtype=torch.float32))
                print("\t\t{}\t{}\n\t\t{}\t{}\n\t\t".format(chrom_file, chrom_len, label_file, label_len))

        data, label = self.removeDuplicated(data, label)

        return data, label
