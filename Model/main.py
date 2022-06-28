from torch.utils.data import random_split as rnd_splt
from chrom_dataset import Chromatin_Dataset



def main():
   chrom_data = Chromatin_Dataset()
   # we are doing a 75/25 split
   train_percent = .75
   train_set, val_set = rnd_splt(chrom_data, [int(chrom_data.length*train_percent), chrom_data.length-int(chrom_data.length*train_percent)])
   print(len(train_set))

main()