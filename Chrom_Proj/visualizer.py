import glob
from sklearn import metrics as m
import matplotlib.pyplot as plt
import tqdm
def processFile(file):
    f = open(file, "r")
    type = f.readline()
    data = f.readline().split(",")
    f.close()

    return data

def combine_coord(location):
    files = glob.glob(location)
    data = {"pre":[],"rec":[],"fpr":[],"tpr":[],}
    for file in files:
        if "pre" in file:
            data["pre"].append((file,processFile(file)))
        if "fpr" in file:
            data["fpr"].append((file,processFile(file)))
        if "rec" in file:
            data["rec"].append((file,processFile(file)))
        if "tpr" in file:
            data["tpr"].append((file,processFile(file)))

    return data


def plotAll(location):
    data = combine_coord(location)
    plt.clf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC Curve')
    
    data["pre"].sort()
    data["rec"].sort()
    
    for pre,rec in zip(data["pre"], data["rec"]):
        print("processing: {}".format(pre[0]))
        print("processing: {}".format(rec[0]))
        pre = pre[1]
        rec = rec[1]
        plt.plot(pre, rec, color="darkgreen")
        print("==================")
    plt.savefig("prc_FULL.png")
    plt.clf()
