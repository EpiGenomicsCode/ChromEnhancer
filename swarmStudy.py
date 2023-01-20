from Chrom_Proj.Swarm_Observer import swarm as Swarm
import numpy as np
import pandas as pd
import glob
import gc
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
    epochs = 20

    # Grab all the models using glob
    files = sorted(glob.glob("./output/model_weight_bias/*MT_1*pt"))
    grav = [.01, .1, .5]
    # for every file in files we want to run a swarm study on it
    for f in files:
        print("Processing: {}".format(f.split("/")[-1]))
        # Load the model
        for G in grav:
            # Folder to save the swarm information
            os.makedirs("./output/Swarm/grav_{}".format(G), exist_ok=True)
            swarm = Swarm.swarm(numParticles, 500, G, 1, 1, f, epochs, f.split("/")[-1])
            swarm.run()
            swarm.save_to_csv("./output/Swarm/grav_{}/{}_Swarm.csv".format(G, f.split("/")[-1]))
            labels = swarm.cluster(5)
            swarm.plot_clusters(labels, "./output/Swarm/grav_{}/{}_Swarm".format(G, f.split("/")[-1]))

        # Delete the model
        del swarm
        gc.collect()
    
if __name__ == "__main__":
    main()