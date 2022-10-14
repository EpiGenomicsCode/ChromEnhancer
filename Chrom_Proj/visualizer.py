from cProfile import label
import glob
from sklearn import metrics as m
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import seaborn as sns


def processFile(file):
    """
        takes in a preprocessed file and returns the data
    """
    f = open(file, "r")
    type = f.readline()
    data = f.readline().strip().split(",")
    f.close()

    return data

def combine_coord(location, name):
    """
        reads in the pre, rec, fpr and tpr data from the file given
    """
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
    labelLocation = (.65, 1.2)
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
        labelName = pre[0][pre[0].index("id_")+3:]
        labelName = labelName.split("_")
        labelName = labelName[0] + " " + labelName[2] + " Model " + labelName[-1][0]
        plt.plot(preConvert, recConvert, label="Model {} AUC:{}".format(labelName, round(m.auc(sorted(preConvert), recConvert),2) ))
        index+=1
        if index == 6:
            index = 0
            plt.legend(bbox_to_anchor=labelLocation)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
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
        labelName = pre[0][pre[0].index("id_")+3:]
        labelName = labelName.split("_")
        labelName = labelName[0] + " " + labelName[2] + " Model " + labelName[-1][0]
        
        plt.plot(fprConvert, tprConvert, label="Model {} AUC:{}".format(labelName, round(m.auc(sorted(fprConvert), tprConvert),2) ))
        index+=1
        if index==6:
            plt.plot([0,1], [0,1],"b--" )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(bbox_to_anchor=labelLocation)
            plt.savefig("output/dataVis/roc_{}_{}.png".format(name,pre[0].split("_")[4]))
            plt.clf()
            index = 0

        print("==================")
    
def plotCluster(plotData, filename, particles):
    totalData = []
    y = ["C1","C2","C3","C4","C5"]
    x = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    plt.clf()
 
    Data = []
    for cluster in plotData.keys():
        clusterData = np.sum(plotData[cluster], 0)
        clusterSections = [
                                sum(clusterData[:100]),
                                sum(clusterData[100:200]),
                                sum(clusterData[200:300]),
                                sum(clusterData[300:400]),
                                sum(clusterData[400:])
                          ]
        Data.append(np.divide(clusterSections, particles))
    
    sns.heatmap(np.array(Data).T/np.linalg.norm(np.array(Data).T), linewidth=.5, xticklabels=x, yticklabels=y, cmap="Spectral") 
    title = filename.split("_")
    grav =title[1]
    title = title[3] + " " + title[4] + " model " + title[-1] + " particles " +  str(particles) 
    
    plt.title(title)
   
    plt.savefig("./output/cluster/grav_{}/{}.png".format(grav, title.replace(" ", "_" )))
    