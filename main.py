from util import dataset as DS
from util import seq_dataset as SeqDS
from util.utils import *
from torch.utils.data import DataLoader
from util.ad_Swarm import swarm as Swarm
import glob
import re


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

    hostname =  os.environ.get("HOSTNAME")
    ids = getids(hostname)
    index = getIndex(hostname)
    epochs = 20
    batch_size = 64
    print("Running on: ", hostname)
    print("Running w/ ids: ", ids)
    print("Running w/ index: ", index)
    seedEverything()
    sequenceStudy(epochs)
    paramatersStudy(ids, index, epochs, batch_size)

def sequenceStudy(epochs=20, batch_size=64):
    trainFiles = glob.glob("Data/230124_CHR-Data_Sequence/CHR-CHROM/TRAIN/*.seq")
    for trainFile in trainFiles:
        name = trainFile[trainFile.rfind("/")+1:-4]

        trainData = SeqDS.Sequence_Dataset(trainFile, type="train")
        trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
        
        testData = SeqDS.Sequence_Dataset(trainFile, type="test")
        testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)

        validData = SeqDS.Sequence_Dataset(trainFile, type="valid")
        validLoader = DataLoader(validData, batch_size=batch_size, shuffle=True)
        # only running on the last model

        model = loadModel(6, name, input_size=4000)
        print(model)
        model = runHomoModel(model, trainLoader, testLoader, validLoader, epochs)
        # run the swarm study
        swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5, size=4000)
                    
        # clear the memory
        clearCache()
        
def getids(hostname):
    """
        Gets the ids for the study
    """
    if "A549" in hostname:
        ids = ["A549"]
    elif "MCF7" in hostname:
        ids = ["MCF7"]
    elif "HepG2" in hostname:
        ids = ["HepG2"]
    elif "K562" in hostname.upper():
        ids = ["K562"]
    else:
        ids =  ["A549" ,"MCF7", "HepG2", "K562"]
    return ids
       
def getIndex(hostname):
    """
        Gets the index for the study
    """
    if "-1" in hostname:
        index = ["-1"]
    elif "-2" in hostname:
        index = ["-2"]
    else:
    index = ["-1", "-2"]
    return index

def paramatersStudy(ids, index, epochs=3, batch_size=64):
    """
        Generates the parameters for the study and runs the study
    """
    

    chromtypes = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    for id in ids:
        for indexType in index: 
            # go through each study
            for study in studys:
                data1, data2 = study.split("-")
                # process the data for each model train, test and test, train
                for data  in [ [data1, data2], [data2, data1]]:
                    train = data[0]
                    test = data[1]

                    
                    ds_train, ds_test, ds_valid = DS.getData([i+indexType for i in chromtypes]  ,id, study, train, test)
                    
                    
                    # cast each dataset to a pytorch dataloader
                    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

                    # go through each model
                    for modelType in range(1,6)[::-1]:
                        name = "id_" + id + "_study_" + study + "_model_" + str(modelType) + "_train_" + train + "_test_" + test + "_type_" + indexType                         

                        log = "\n\tid: {}\n\tstudy: {}\n\tmodel: {}\n\ttrain: {}\n\ttest: {}\n\ttype: {}\n\t".format(id, study, modelType, train, test, indexType)
                        
                        print(name)
                        print(log)
                        print("\t\ttrain label: {}".format(ds_train.labelFile))
                        print("\t\ttest label: {}".format(ds_test.labelFile))
                        print("\t\tvalid label: {}".format(ds_valid.labelFile))
                
                        # run the model
                        model = loadModel(modelType, name)
                        print(model)
                        model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                        
                        # run the swarm study
                        swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5)
                    
                        # clear the memory
                        clearCache()

def swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5, size=500):
    swarm = Swarm.swarm(num_particles, gravity,  epochs, model, size=size)
    swarm.run()
    # get all particles in swarm
    particles = np.array([i.position.numpy() for i in swarm.swarm])
    particles = particles.reshape(particles.shape[0], particles.shape[2])

    # cluster the particles with sklearn hierarchical clustering
    cluster = clusterParticles(particles, 5)

    # plot the clusters
    plotClusters(cluster, particles, name)

                       
if __name__ == "__main__":
    main()