from Chrom_Proj.chrom_dataset import readFiles
from Chrom_Proj.util import *
import glob
from sklearn import metrics as m
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
from sklearn import metrics as m
import torch
from torchviz import make_dot
import tensorflow as tf
import os

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
    title = title[3] + " " + title[4] + " model " + title[-2] + " particles " +  str(particles) 
    
    plt.title(title)
    os.makedirs("./output/Swarm/grav_{}".format(grav), exist_ok=True)

    plt.savefig("./output/Swarm/grav_{}/{}.png".format(grav, title.replace(" ", "_" )))

def plotPRC(model, pre, rec):
    os.makedirs("./output/prc/", exist_ok=True)

    """
        plots the PRC curve:
        
        input:
        ======
            model: pytorch model

            pre: array: precisison data

            rec: array: recall data
    """

    prc_auc = m.auc(sorted(pre), rec)

    plt.clf()
    plt.plot(pre, rec, color="darkgreen", 
            label='PRC curve (area = %0.2f' % prc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PRC Curve')
    plt.legend(loc="lower right")
    plt.savefig("output/prc/{}_prc.png".format(model.name))
    plt.clf()

    return prc_auc

def plotROC(model, fpr, tpr):
    os.makedirs("./output/roc/", exist_ok=True)

    """
        plots the ROC curve:
        
        input:
        ======
            model: pytorch model

            fpr: array: false positive rate

            tpr: array: true positive rate
    """
    roc_auc = m.auc(fpr, tpr)
    plt.clf()
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("output/roc/{}_roc.png".format(model.name))

    return roc_auc
    


def plotModel():
    modelTypes = [1,2,3]
    os.makedirs("./output/dataVis/modelGraphs/", exist_ok=True)
    for modelType in modelTypes:
        print("generating graph for model type: {}".format(modelType))
        model = loadModel(modelType, str(modelType))
        ranData = torch.rand(500,500)
        yhat = model(ranData)

        make_dot(yhat,params=dict(list(model.named_parameters()))).render("output/dataVis/modelGraphs/"+str(modelType)+"_visualizer", format="png")


def modelPreformance():
    modelType = [1,2,3,4,5,6]
    os.makedirs("./output/dataVis/BW_ROC/", exist_ok=True)
    os.makedirs("./output/dataVis/BW_PRC/", exist_ok=True)
    cellLine = ["A549", "HepG2", "K562", "MCF7"]
    totalDataROC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
    totalDataPRC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
    for cl in cellLine:
        clDataROC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
        clDataPRC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
        for mt in modelType:
            for fileName in glob.glob("output/Info/Analysis_id_{}*MT_{}*".format(cl, mt)):
                data = readAnalysisData(fileName)
                clDataROC["model {}".format(mt)].append(data["ROCAUC"])
                clDataPRC["model {}".format(mt)].append(data["PRCAUC"])
        
        for mt in modelType:
            totalDataROC["model {}".format(mt)].append(clDataROC["model {}".format(mt)])
            totalDataPRC["model {}".format(mt)].append(clDataPRC["model {}".format(mt)])
        
        fig, ax = plt.subplots()
        plt.title("Average AUROC for {}".format(cl))
        ax.boxplot(clDataROC.values())
        ax.set_xticklabels(clDataROC.keys())
        plt.savefig("./output/dataVis/BW_ROC/{}.png".format(cl))

        plt.clf()

        fig, ax = plt.subplots()
        plt.title("Average AUPRC for {}".format(cl))
        ax.boxplot(clDataPRC.values())
        ax.set_xticklabels(clDataPRC.keys())
        plt.savefig("./output/dataVis/BW_PRC/{}.png".format(cl))

        plt.clf()

            

    # average of total
    fig, ax = plt.subplots()
    plt.title("Average AUROC for all CL")
    # we need to average these values
    data = []
    for key in totalDataROC.keys():
        data.append(np.mean(totalDataROC[key], axis=0))
    ax.boxplot(data)
    ax.set_xticklabels(totalDataROC.keys())
    plt.savefig("./output/dataVis/BW_ROC/Ave.png")
    plt.clf()

    fig, ax = plt.subplots()
    plt.title("Average AUPRC for all CL")
    # we need to average these values
    data = []
    for key in totalDataPRC.keys():
        data.append(np.mean(totalDataPRC[key], axis=0))
    ax.boxplot(data)
    ax.set_xticklabels(totalDataPRC.keys())
    plt.savefig("./output/dataVis/BW_PRC/Ave.png")
    plt.clf()
    
def modelLoss():
    os.makedirs("./output/dataVis/Loss/", exist_ok=True)

    files = glob.glob("output/Info/Loss*")
    for fileName in files:
        plt.title(fileName)
        data = readLossData(fileName)
        for key, value  in data.items():
            plt.plot(value, label=str(key))
        plt.legend()
        plt.savefig("./output/dataVis/Loss/{}.png".format(fileName[fileName.rfind("/"):-4]))
        plt.clf()
            
