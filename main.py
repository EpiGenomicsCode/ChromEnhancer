from util import dataset as DS
from util.utils import *
from torch.utils.data import DataLoader
from util.ad_Swarm import swarm as Swarm
import gc

def seedEverything(seed=42):
    """
        Seeds everything for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seedEverything()
    paramatersStudy()
    
def paramatersStudy():
    """
        Generates the parameters for the study and runs the study
    """
    ids = ["MCF7"]
    chromtypes = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    epochs = 20
    batch_size = 2048
    for indexType in ["-1"]:
        # CTCF-1 vs CTCF-2
        chromtypes = [i+indexType for i in chromtypes]
        # go through each id
        for id in ids:
            # go through each study
            for study in studys:
                # go through each model
                for modelType in range(0,6):
                    data1, data2 = study.split("-")
                    # process the data for each model train, test and test, train
                    for data  in [ [data1, data2], [data2, data1]]:
                        train = data[0]
                        test = data[1]

                        ds_train, ds_test, ds_valid = DS.getData(chromtypes,id, study, train, test)
                        
                        # print the parameters
                        print("id: ", id)
                        print("\tstudy: ", study)
                        print("\tmodel: ", modelType)
                        print("\ttrain:", train )
                        print("\ttest:", test)
                        
                        # cast each dataset to a pytorch dataloader
                        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                        valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

                        # run the model
                        name = "id_" + id + "_study_" + study + "_model_" + str(modelType) + "_train_" + train + "_test_" + test + "_type_" + indexType                         
                        print(name)
                        model = loadModel(modelType, name)
                        model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                        
                        # run the swarm study
                        swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5)
                       
                        # clear the memory
                        clearCache()


def swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5):
    swarm = Swarm.swarm(num_particles, gravity,  epochs, model)
    swarm.run()
    # get all particles in swarm
    particles = np.array([i.position.numpy() for i in swarm.swarm])
    particles = particles.reshape(particles.shape[0], particles.shape[2])

    # cluster the particles with sklearn hierarchical clustering
    cluster = clusterParticles(particles, 5)

    # plot the clusters
    plotClusters(cluster, particles, name)


# clear the cache and gpu memory
def clearCache():
    torch.cuda.empty_cache()
    gc.collect()
                        
if __name__ == "__main__":
    main()