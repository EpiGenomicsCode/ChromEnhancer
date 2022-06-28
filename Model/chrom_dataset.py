from torch.utils.data import Dataset
from glob import glob
import numpy as np

class Chromatin_Dataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.data, self.labels = self.readfiles()
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        assert len(self.data) == len(self.labels)

        self.length = len(self.data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def readfiles(self, file_location="../Data/Bichrom_sample_data/"):
        chrom_filenames = glob(file_location+"*chrom*")
        label_filenames = glob(file_location+"*label*")

        data  = []
        label = []
        print("Processing\n\tchrome files {}\n\tlabel files {}".format(chrom_filenames, label_filenames))
        for files in zip(chrom_filenames, label_filenames):
            chrom_file = files[0]
            label_file = files[1]
            if "val" not in chrom_file:
                chrom_len = 0
                label_len = 0
                with open(chrom_file) as infile:
                    for line in infile:
                        chrom_len+=1
                        data.append([float(i) for i in line.split()])
                with open(label_file)  as infile:
                    for line in infile:
                        label_len+=1
                        label.append(int(line))
                print("\t\t{}\t{}\n\t\t{}\t{}".format(chrom_file, chrom_len, label_file, label_len))


        return data, label
