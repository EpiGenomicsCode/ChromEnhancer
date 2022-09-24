import glob
from sklearn import metrics as m
import matplotlib.pyplot as plt
import pandas as pd

def processFile(file):
    f = open(file, "r")
    type = f.readline()
    data = f.readline().strip().split(",")
    f.close()

    return data

def combine_coord(location, name):
    print("looking at {}".format("location"))
    files = glob.glob(location+"*{}*".format(name))
    print("found files: {}".format(files))
    assert len(files) !=0
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

def plotAll(location, name):
    plt.rcParams["figure.figsize"] = [15.00, 10.00]
    plt.rcParams["figure.autolayout"] = True
    data = combine_coord(location,name)
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
    index = 0
 
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

        plt.plot(preConvert, recConvert, label="Model {}_AUC:{}".format(pre[0].split("_")[-1:][0], round(m.auc(sorted(preConvert), recConvert),2) ))
        index+=1
        if index == 5:
            index = 0
            plt.plot([0,1], [0,1])
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.savefig("output/dataVis/prc_{}_{}.png".format(name, pre[0].split("_")[4]))
            plt.clf()
        print("==================")
    
    #==============================
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    index = 0
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
        plt.plot(fprConvert, tprConvert, label="Model {}_AUC:{}".format(pre[0].split("_")[-1][0], round(m.auc(sorted(fprConvert), tprConvert),2) ))
        index+=1
        if index==5:
            plt.plot([0,1], [0,1])
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            plt.savefig("output/dataVis/roc_{}_{}.png".format(name,pre[0].split("_")[4]))
            plt.clf()
            index = 0

        print("==================")
    
