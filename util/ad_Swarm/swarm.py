import numpy as np
from . import particle
import copy
import torch
import pdb

class swarm:
    def __init__(self, num_particles, gravity,  epochs, model, size=500):
        self.swarm = []
        self.epochs = epochs
        device = "cpu"
        self.model = model.to(device)
        self.gravity = gravity
        for i in range(num_particles):
            self.swarm.append(particle.particle(size))
        self.total_mass = 0
        self.gbest = None
        self.gworse = None
        self.solution = None
        self.calcMass()

    def calcMass(self):
        self.total_mass = 0
        for particle in self.swarm:
            particle.score = self.model(particle.position).item()


        for particle in self.swarm:
            if self.gbest == self.gworse == self.solution == None:
                self.gbest = self.gworse = self.solution = particle

            if self.gbest.score > particle.score:
                self.gbest = copy.deepcopy(particle)

            if self.solution.score > particle.score:
                # print("Old solution found:{}".format(str(self.solution)))
                self.solution = copy.deepcopy(particle)
                # print("New solution found:{}".format(str(self.solution)))

            if self.gworse.score < particle.score:
                self.gworse = copy.deepcopy(particle)


    def norMass(self):
        for particle in self.swarm:
            particle.mass = (particle.score - self.gworse.score +.01)
            particle.mass /= (self.gbest.score - self.gworse.score)+.01
            self.total_mass += particle.mass

        for particle in self.swarm:
            particle.mass /= self.total_mass


    def calcForce(self, iteration):
        for particle in self.swarm:
            forces = []
            for otherParticle in self.swarm:
                if particle != otherParticle:

                    force = self.gravity * np.e**(-iteration)+ len(self.swarm)
                    force *= (particle.mass * otherParticle.mass)
                    force /= (np.random.rand()+np.linalg.norm(np.subtract(particle.position,otherParticle.position))**2)
                    force *= np.subtract(otherParticle.position,particle.position)
                    forces.append(force)
            particle.force = sum(forces)

    def moveForce(self):
        for particle in self.swarm:
            particle.force /= particle.mass
            particle.force = np.multiply(np.random.rand() ,particle.force)
            particle.position +=  particle.force
            particle.position = [max(0,i) for i in particle.position[0].tolist()]
            particle.position = [min(1,i) for i in particle.position]
            particle.position = torch.tensor([list(map(abs, particle.position))], dtype=torch.float32)

            particle.history.append(torch.abs(particle.position))
        

    def run(self):
        for i in range(self.epochs):
            self.calcMass()
            self.norMass()
            self.calcForce(i)
            self.moveForce()            
        self.calcMass()
        
        



    
    def __str__(self):
        info = ""
        for p in self.swarm:
            info += str(p.position[0][:3]) + "\t"+ str(p.score)+"\n"

        return info + "\n"