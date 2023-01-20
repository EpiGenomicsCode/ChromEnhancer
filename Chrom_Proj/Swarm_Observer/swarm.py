import random
import tqdm 
from Chrom_Proj.util import loadModelfromFile
import torch
import numpy as np
import csv
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

class Particle:
    """
    A particle in the swarm
    args:
        position: the position of the particle
        velocity: the velocity of the particle
        best_position: the best position of the particle
    """
    def __init__(self, position, velocity, best_position):
        self.position = position
        self.velocity = velocity
        self.best_position = best_position

        
class swarm:
    def __init__(self, num_particles, num_dimensions, G, a, b, filename, iterations, name):
        """ 
        Initialize the swarm
            Args:
                num_particles: the number of particles in the swarm
                num_dimensions: the number of dimensions in the swarm
                G: the gravitational pull
                a: the local pull
                b: the global pull
                filename: the filename of the model weights and bias
                iterations: the number of iterations to run
                name: the name of the swarm
        """
        # save the parameters
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.G = G
        self.a = a
        self.b = b
        self.swarm = []
        self.iterations = iterations
        self.model = loadModelfromFile(filename, int(filename[-4]))
        self.name = name
        # initialize the swarm
        for i in range(num_particles):
            # we initalize a random position
            position = [random.uniform(-1, 1) for _ in range(num_dimensions)]
            # we have a random velcotiy
            velocity = [random.uniform(-1, 1) for _ in range(num_dimensions)]

            # initialize the best position randomly
            best_position = position

            # save the particle
            self.swarm.append(Particle(position, velocity, best_position))
        
        # Evaluate the global best position.
        self.global_best_position = min(self.swarm, key=lambda p: self.evaluate(p.position))

    def evaluate(self, position):
        return self.model(torch.tensor(np.array(position), dtype=torch.float32)).item()

    def update(self):
        """ 
        Update the swarm
        """
        for particle in self.swarm:
            # update each particles position WRT the dimension and graviational pull
            for i in range(self.num_dimensions):
                particle.velocity[i] += self.a * random.uniform(0, 1) * (particle.best_position[i] - particle.position[i])
                particle.velocity[i] += self.b * random.uniform(0, 1) * (self.global_best_position.position[i] - particle.position[i])
                particle.position[i] += self.G * particle.velocity[i]
            # update the particles position
            evaluation = self.evaluate(particle.position)

            # find the new global and local best
            if evaluation < self.evaluate(particle.best_position):
                particle.best_position = particle.position
            if evaluation < self.evaluate(self.global_best_position.position):
                self.global_best_position = particle

    def save_to_csv(self, filename):
        """
        Save the swarm to a csv file
            args:
                filename: the filename to save the swarm to
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pos_{}".format(i) for i in range(self.num_dimensions)])  # 
            for particle in self.swarm:
                writer.writerow(particle.position)

    def cluster(self, num_clusters):
        """
        Cluster the swarm using hierarchical clustering
            args:
                num_clusters: the number of clusters to use
            
        """
        positions = np.array([particle.position for particle in self.swarm])
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
        clustering.fit(positions)
        return clustering.labels_
 
    def plot_clusters(self, labels, filename):
        """
        Plot the clusters
            args:
                labels: the labels of the clusters
                filename: the filename to save the plot to
        """
        data = {}
        positions = np.array([particle.position for particle in self.swarm])
        for i in zip(positions, labels):
            if i[1] not in data:
                data[i[1]] = []
            data[i[1]].append(i[0])
        
        x = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
        y = ["Cluster_{}".format(i) for i in range(0, max(labels))]


        plotData = []
        # Plots the mean and the mean of the clusters based on their individual sections.
        for i in range(max(labels)):
            averageCluster = np.mean(data[i], axis=0)
            clusterSection =  [
                                sum(averageCluster[:100]),
                                sum(averageCluster[100:200]),
                                sum(averageCluster[200:300]),
                                sum(averageCluster[300:400]),
                                sum(averageCluster[400:])
                          ]
            plotData.append(clusterSection)
        sns.heatmap(plotData, xticklabels=x, yticklabels=y, linewidths=.5)
        print("saving to {}".format(filename+".png"))
        plt.savefig(filename+".png")
        plt.clf()

    def run(self):
        for i in tqdm.tqdm(range(self.iterations), desc=self.name):
            self.update()
        