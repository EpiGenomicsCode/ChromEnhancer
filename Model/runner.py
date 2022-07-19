from Model.chrom_dataset import Chromatin_Dataset

def getK562(chromtypes):
    K562_chr10_17 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr10-chr17",
        file_location="./Data/220708_DATA/TRAIN/*") # 499100

    K562_chr10 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr10",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    K562_chr17 = Chromatin_Dataset(
        id="K562",
        chromType=chromtypes,
        label="chr17",
        file_location="./Data/220708_DATA/HOLDOUT/*")

    trainer = [K562_chr10_17]
    tester = [K562_chr10]
    validator = [K562_chr17]

    return trainer, tester, validator