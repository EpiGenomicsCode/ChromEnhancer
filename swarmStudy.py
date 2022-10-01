from Chrom_Proj.Swarm_Observer import swarm
from Chrom_Proj.runner import loadModel
import pdb
import numpy as np
import pandas as pd
import glob
import gc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from Chrom_Proj.visualizer import plotCluster

def main():
    n_clusters = 5
    numParticles = 10
    files = sorted(glob.glob("./output/model_weight_bias/*epoch_10*pt"))
    for f in files:
        print("Processing: {}".format(f.split("/")[-1]))
        s, model = swarmModel(modelLocation=f, modelType=int(f[-4]),numParticles=numParticles
                                ,gravity=0,epochs=10)
        saveOutput(s, model,  f[:-3]+"_Swarm.csv")
        plotData = clusterSwarm(s, n_clusters)
        print("saving clusters to {}".format(f[:-3]))
        plotCluster(plotData, "output/cluster/{}".format(f[f.rindex("/")+1:-3]))
        del model
        gc.collect()

def swarmModel(modelLocation="./output/model_weight_bias/model_id_A549_TTV_chr10-chr17_chr10_chr17_epoch_10_BS_32_FL_-Data-220802_DATA_MT_1.pt"
            , modelType=1, numParticles=10, gravity=10, epochs=10):
    model = loadModel(modelFileName=modelLocation, modelType=modelType)
    model.eval()
    s = swarm.swarm(numParticles, gravity,  epochs, model)
    s.run()
    return s, model

def saveOutput(swarm, model, fileName="swarmOutput.csv"):
    s = swarm
    data = []
    for particle in s.swarm:
        history = particle.history[-1].tolist()
        history[0].append(model(particle.history[-1]).item())
        data.append(history[0])


    data = pd.DataFrame(data)
    data = data.round(4)
    data.to_csv("./output/swarm/"+fileName[fileName.rfind("/")+1:], index=False, header=False)

    plt.clf()
    for index,row in data.iterrows():
        plt.plot(list(row))
    plt.savefig("./output/swarm/"+fileName[fileName.rfind("/")+1:-3]+".png")


def clusterSwarm(swarm, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    positions = np.array([birb.position.tolist() for birb in swarm.swarm])
    positions = np.squeeze(positions, axis=1)
    hc = hc.fit(positions)
    predictedLabels = hc.labels_
    plotData = {}
    for comb in zip(positions, predictedLabels):
        if not comb[1] in plotData.keys():
            plotData[comb[1]] = [comb[0]]
        else:
            plotData[comb[1]].append(comb[0])

    return plotData







main()

