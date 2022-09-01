from Chrom_Proj.Swarm_Observer import swarm
from Chrom_Proj.runner import loadModel

def main():
    model = loadModel(modelFileName="./output/model_weight_bias/model_id_A549_TTV_chr10-chr17_chr10_chr17_epoch_10_BS_32_FL_-Data-220802_DATA_MT_1.pt", modelType=1)
    num_particles = 5
    gravity = 10
    epochs = 5
    s = swarm.swarm(num_particles, gravity,  epochs, model)
    print(s)
    s.run()


main()