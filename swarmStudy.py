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
import os

def main():
    """
    performs a swarm study on each of the models input to extract features 
    from the overall network

    Variables:
        numClusters: the amount of clusters to use for all particles
        numParticles: the number of particles to use in the test
    """
    numClusters = 5
    numParticles = 100
    epochs = 10

    # Grab all the weights and biases saved
    files = sorted(glob.glob("./output/model_weight_bias/*pt"))
    grav = [.5, .05, .005]
    for g in grav:
        print("Grav: {}".format(g))
        for f in files:
            print("Processing: {}".format(f.split("/")[-1]))

            # Save the plots and generate a heatmap of the clusters
            if not os.path.exists("./output/cluster/grav_{}".format(g)):
                os.mkdir("./output/cluster/grav_{}".format(g))
                
            # performs the study 
            s, model = swarmModel(modelLocation=f, modelType=int(f[-4]),numParticles=numParticles
                                    ,gravity=g,epochs=epochs)
            # save the particles WRT their model
            saveOutput(s, model,  "./output/swarm/", f[:-3]+"_Swarm.csv")

            # Cluster the swarm into different sections
            plotData = clusterSwarm(s, numClusters)
            print("saving clusters to {}".format(f[:-3]))

                
            plotCluster(plotData, "output/cluster/grav_{}_/{}".format(g, f[f.rindex("/")+1:-3]), numParticles)
            
            # Clean up
            del model
            gc.collect()

def swarmModel(modelLocation="./output/model_weight_bias/model_id_A549_TTV_chr10-chr17_chr10_chr17_epoch_10_BS_32_FL_-Data-220802_DATA_MT_1.pt"
                ,modelType=1
                ,numParticles=10,
                gravity=10,
                epochs=10):
    """
    Perform a GSA algorithm on the input to extract features that guarentee an enhancer 

    Input:
        modelLocation: relative path of where the model weight and bias are saved
        modelType: the model architecture (see model in Chrom folder)
        numParticles: the number of particles in the swarm study
        gravity: the gravitational pull for communication in the GSA swarm
        epochs: number of epochs in the swarm study
    
    Return:
        swarm: optimized swarm
        model: loaded model
    """

    # load the model
    model = loadModel(modelLocation, modelType)
    
    # we do not need the gradients just evaluation
    model.eval()

    # Generate a Swarm
    s = swarm.swarm(numParticles, gravity,  epochs, model)
    # Run the swarm
    s.run()
    
    # return the swarm and associated model
    return s, model

def saveOutput(swarm, 
                model, 
                saveLocation="./output/swarm/",
                fileName="swarmOutput.csv"):
    """
    Given an optimized swarm and corosponding model 
    save the position of the swarm as a CSV file 
    where the first 500 rows are the positions and the last one is the predicted value
    
    Input:
        swarm: swarm that was run on a specific model
        model: model the swarm was run on
        saveLocation: Location to save the 
    """
    s = swarm
    data = []
    # get the final position of the swarm
    for particle in s.swarm:
        history = particle.history[-1].tolist()
        score = model(particle.history[-1]).item()
        if score > .5:
            history[0].append(score)
            data.append(history[0])

    # Save the data as a CSV rounded to 4th decimal
    data = pd.DataFrame(data)
    data = data.round(4)
    print("saving to {}".format(saveLocation+fileName[fileName.rfind("/")+1:]))
    data.to_csv(saveLocation+fileName[fileName.rfind("/")+1:], index=False, header=False)
    
def clusterSwarm(swarm, numClusters):
    """ 
    cluster the particles in the swarm

    input:
        swarm: swarm object that has ran on a model
        numCluster: the number of clusters you want to use for the swarm

    returns:
        plotData: 2-D array of the data and the cluster it belongs to
    """
    # we are using hiarachacle clustering 
    hc = AgglomerativeClustering(numClusters, linkage="ward")
    
    # get the positions of the particles in the swarm
    positions = np.array([birb.position.tolist() for birb in swarm.swarm])
    positions = np.squeeze(positions, axis=1)
    # fit the particles to a cluster
    hc = hc.fit(positions)

    # get the predicted cluster labels
    predictedLabels = hc.labels_
    plotData = {}

    # map each particle to its cluster
    for comb in zip(positions, predictedLabels):
        if not comb[1] in plotData.keys():
            plotData[comb[1]] = [comb[0]]
        else:
            plotData[comb[1]].append(comb[0])

    return plotData


main()

