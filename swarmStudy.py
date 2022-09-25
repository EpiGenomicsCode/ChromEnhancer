from Chrom_Proj.Swarm_Observer import swarm
from Chrom_Proj.runner import loadModel
import pdb
import numpy as np
import pandas as pd
import glob
import gc
import matplotlib.pyplot as plt

def main():
    files = sorted(glob.glob("./output/model_weight_bias/*epoch_10*pt"))
    for f in files:
        print("Processing: {}".format(f.split("/")[-1]))
        s, model = swarmModel(modelLocation=f, modelType=int(f[-4]),numParticles=100,gravity=0,epochs=10)
        saveOutput(s, model,  f[:-3]+"_Swarm.csv")
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

main()

