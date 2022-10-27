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
   
    plt.savefig("./output/Swarm/grav_{}/{}.png".format(grav, title.replace(" ", "_" )))

def plotPRC(model, pre, rec):
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
    


def plotModel(modelType):
    print("generating graph for model type: {}".format(modelType))
    model = loadModel(modelType, str(modelType))
    ranData = torch.rand(500,500)
    yhat = model(ranData)

    make_dot(yhat,params=dict(list(model.named_parameters()))).render("output/dataVis/modelGraphs/"+str(modelType)+"_visualizer", format="png")


def modelPreformance(modelType, cellLine, fileLocation="./output/Info/"):
    modelType = [1,2,3,4,5,6]
    cellLine = ["A549", "HepG2", "K562", "MCF7"]
    totalDataROC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
    totalDataPRC = {"model 1":[],"model 2":[],"model 3":[],"model 4":[],"model 5":[],"model 6":[]}
    for cl in cellLine:
        for mt in modelType:
            for fileName in glob.glob("output/Info/Analysis_id_{}*MT_{}*".format(cl, mt)):
                data = readData(fileName)
            
                totalDataROC["model {}".format(mt)].append(data["ROCAUC"])
                totalDataPRC["model {}".format(mt)].append(data["ROCAUC"])
    
    
        fig, ax = plt.subplots()
        plt.title("Average AUROC for {}".format(cl))
        ax.boxplot(totalDataROC.values())
        ax.set_xticklabels(totalDataROC.keys())
        plt.savefig("./output/dataVis/BW_ROC/{}.png".format(cl))

        plt.clf()

        fig, ax = plt.subplots()
        plt.title("Average AUPRC for {}".format(cl))
        ax.boxplot(totalDataPRC.values())
        ax.set_xticklabels(totalDataPRC.keys())
        plt.savefig("./output/dataVis/BW_PRC/{}.png".format(cl))

        plt.clf()

        