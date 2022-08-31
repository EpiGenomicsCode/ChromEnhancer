import glob
from sklearn import metrics as m
import matplotlib.pyplot as plt
import tqdm


def processFile(file):
    f = open(file, "r")
    type = f.readline()
    data = f.readline().strip().split(",")
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

    data["pre"].sort()
    data["rec"].sort()
    data["fpr"].sort()
    data["tpr"].sort()
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC Curve')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    for pre,rec in zip(data["pre"], data["rec"]):
        print("processing: {}".format(pre[0]))
        print("processing: {}".format(rec[0]))
        
        preConvert = []
        recConvert = []
        for i in pre[1]:
            try:
                preConvert.append(float(i))
            except:
                continue
        for i in rec[1]:
            try:
                recConvert.append(float(i))
            except:
                continue
        plt.plot(preConvert, recConvert, label="CH:{}_Model:{}_AUC:{}".format( pre[0].split("_")[2], pre[0].split("_")[-1][0], round(m.auc(sorted(preConvert), recConvert),2) ))
        print("==================")
    plt.plot([0,1], [0,1])
    plt.legend()
    plt.savefig("prc_FULL.png")
    plt.clf()
    #==============================, 
    for pre,rec in zip(data["fpr"], data["tpr"]):
        print("processing: {}".format(pre[0]))
        print("processing: {}".format(rec[0]))
        
        fprConvert = []
        tprConvert = []
        for i in pre[1]:
            try:
                fprConvert.append(float(i))
            except:
                continue
        for i in rec[1]:
            try:
                tprConvert.append(float(i))
            except:
                continue
        plt.plot(fprConvert, tprConvert, label="CH:{}_Model:{}_AUC:{}".format( pre[0].split("_")[2], pre[0].split("_")[-1][0], round(m.auc(sorted(fprConvert), tprConvert),2) ))
        print("==================")
    plt.plot([0,1], [0,1])
    plt.legend()
    plt.savefig("roc_FULL.png")
    plt.clf()

# def vizData():
#     trainer, tester, validator = getData(chromtypes="A549",  
#                             id=id, 
#                             trainLabel="chr10-chr17", 
#                             testLabel="chr10", 
#                             validLabel="chr17",
#                             fileLocation="./Data/220802_DATA")
