import matplotlib.pyplot as plt
import pdb

chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
l1 = []
for ch in chromtypes:
    x = open("./Data/220802_DATA/TRAIN/A549_chr10-chr17_train_{}_combined.chromtrack".format(ch))
    l1.append([float(k) for k in x.readline().split()])
    x.close()

pdb.set_trace()
