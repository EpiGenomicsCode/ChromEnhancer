from util import dataset as DS
from util import seq_dataset as SeqDS
from util.utils import *
from torch.utils.data import DataLoader
from util.ad_Swarm import swarm as Swarm
import glob

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
    sequenceStudy()
    paramatersStudy()

def sequenceStudy():
    epochs = 20
    trainFiles = glob.glob("Data/230124_CHR-Data_Sequence/CHR-CHROM/TRAIN/*.seq")

    for trainFile in trainFiles:
        name = trainFile[trainFile.rfind("/")+1:-4]

        trainData = SeqDS.Sequence_Dataset(trainFile, type="train")
        trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
        
        testData = SeqDS.Sequence_Dataset(trainFile, type="test")
        testLoader = DataLoader(testData, batch_size=64, shuffle=True)

        validData = SeqDS.Sequence_Dataset(trainFile, type="valid")
        validLoader = DataLoader(validData, batch_size=64, shuffle=True)
        
        for i in range(1,7):
            model = loadModel(i, name, input_size=4000)
            model = runHomoModel(model, trainLoader, testLoader, validLoader, epochs)
            # run the swarm study
            swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5, size=4000)
                        
            # clear the memory
            clearCache()

    
def paramatersStudy():
    """
        Generates the parameters for the study and runs the study
    """
    ids =  ["A549" ,"MCF7", "HepG2", "K562"]
    chromtypes = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    epochs = 20
    batch_size = 64
    for id in ids:
        for indexType in ["-1", "-2"]: 
            # go through each study
            for study in studys:
                # go through each model
                for modelType in range(1,7):
                    data1, data2 = study.split("-")
                    # process the data for each model train, test and test, train
                    for data  in [ [data1, data2], [data2, data1]]:
                        train = data[0]
                        test = data[1]
                        name = "id_" + id + "_study_" + study + "_model_" + str(modelType) + "_train_" + train + "_test_" + test + "_type_" + indexType                         

                        log = "\n\tid: {}\n\tstudy: {}\n\tmodel: {}\n\ttrain: {}\n\ttest: {}\n\ttype: {}\n\t".format(id, study, modelType, train, test, indexType)
                        
                        ds_train, ds_test, ds_valid = DS.getData([i+indexType for i in chromtypes]  ,id, study, train, test)
                        print(name)
                        print(log)
                        print("\t\ttrain label: {}".format(ds_train.labelFile))
                        print("\t\ttest label: {}".format(ds_test.labelFile))
                        print("\t\tvalid label: {}".format(ds_valid.labelFile))
                
                        # cast each dataset to a pytorch dataloader
                        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                        valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

                        # run the model

                        model = loadModel(modelType, name)
                        model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                        
                        # run the swarm study
                        swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5)
                    
                        # clear the memory
                        clearCache()


def swarmStudy(model, name, epochs=10, num_particles=10, gravity=.5, size=500):
    swarm = Swarm.swarm(num_particles, gravity,  epochs, model, size)
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