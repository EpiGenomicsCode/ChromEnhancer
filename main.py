from util import dataset as DS
from util import seq_dataset as SeqDS
from util.utils import *
from torch.utils.data import DataLoader
import glob
import re
# add command line arguments
import argparse
from itertools import combinations


# add command line arguments
parser = argparse.ArgumentParser(description='Run the study')
parser.add_argument('--sequence', action='store_true', help='Run the sequence study')
parser.add_argument('--parameter', action='store_true', help='Run the parameter study')
parser.add_argument('--parameterCLD', action='store_true', help='Run the parameter study with command line dropout')

# optional ids
parser.add_argument('--ids', nargs='+', help='Run the study on the ids', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--index', nargs='+', help='Run the study on the index', default=["-1","-2"])
parser.add_argument('--model', nargs='+', help='Run the study on the model', default=["1", "2", "3", "4", "5"])
parser.add_argument('--batch_size', type=int, help='Run the study on the batch size', default=32)
parser.add_argument('--epochs', type=int, help='Run the study on the epochs', default=20)

args = parser.parse_args()

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
    # if ids are not specified, get the ids
    
    if not parser.parse_args().ids:
        ids = getids(hostname)
    else:
        ids = args.ids

    if not args.ids:
        index = getIndex(hostname)
    else:
        index = args.index

    epochs = args.epochs
    batch_size = args.batch_size
    print("Running on: ", hostname)
    print("Running w/ ids: ", ids)
    print("Running w/ index: ", index)
    seedEverything()


    if args.sequence:
        sequenceStudy(epochs, batch_size)

    if args.parameter:
        paramatersStudy(ids, index, epochs, batch_size)

    if args.parameterCLD:
        paramatersIndependentStudy(ids, index, epochs, batch_size)

def sequenceStudy(epochs=20, batch_size=64):
    trainFiles = glob.glob("./Data/230124_CHR-Data_Sequence/CHR-CHROM/TRAIN/*.seq")
    for trainFile in trainFiles:
        name = trainFile[trainFile.rfind("/")+1:-4]

        trainData = SeqDS.Sequence_Dataset(trainFile, type="train")
        trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
        
        testData = SeqDS.Sequence_Dataset(trainFile, type="test")
        testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)

        validData = SeqDS.Sequence_Dataset(trainFile, type="valid")
        validLoader = DataLoader(validData, batch_size=batch_size, shuffle=True)
        for i in args.model[::-1]:
            name = "seq_"+name + "_model{}".format(i)
            model = loadModel(i, name, input_size=4000)
            print(name)
            print(model)
            model = runHomoModel(model, trainLoader, testLoader, validLoader, epochs)

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

                    for modelType in args.model[::-1]:
                        name = "param_id_" + id + "_study_" + study + "_model_" + str(modelType) + "_train_" + train + "_test_" + test + "_type_" + indexType                         
                        log = "\n\tid: {}\n\tstudy: {}\n\tmodel: {}\n\ttrain: {}\n\ttest: {}\n\ttype: {}\n\t".format(id, study, modelType, train, test, indexType)
                        
                        ds_train, ds_test, ds_valid = DS.getData([i+indexType for i in chromtypes]  ,[id], study, train, test)
                        
                        # cast each dataset to a pytorch dataloader
                        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                        valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)
                        print(name)
                        print(log)
                
                        # run the model
                        model = loadModel(modelType, name)
                        print(model)
                        model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                        
                    
                        # clear the memory
                        clearCache()

def paramatersIndependentStudy(ids, index, epochs=3, batch_size=64):
    """
        Generates the parameters for the study and runs the study
    """
    chromtypes = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]
    
    for id in combinations(ids, len(ids)-1):
        id = list(id)
        for indexType in index: 
            # go through each study
            for study in studys:
                data1, data2 = study.split("-")
                # process the data for each model train, test and test, train
                for data  in [ [data1, data2], [data2, data1]]:
                    train = data[0]
                    test = data[1]

                    # go through each model
                    for modelType in args.model[::-1]:
                        name = "CLD_id_" + '-'.join(id) + "_study_" + study + "_model_" + str(modelType) + "_train_" + train + "_test_" + test + "_type_" + indexType                         
                        log = "\n\tid: {}\n\tstudy: {}\n\tmodel: {}\n\ttrain: {}\n\ttest: {}\n\ttype: {}\n\t".format('-'.join(id), study, modelType, train, test, indexType)
                        
                        ds_train, ds_test, ds_valid = DS.getData([i+indexType for i in chromtypes]  ,id, study, train, test)
                        
                        
                        # cast each dataset to a pytorch dataloader
                        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                        valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)
                        print(name)
                        print(log)
                       
                
                        # run the model
                        model = loadModel(modelType, name)
                        print(model)
                        model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                        
                    
                        # clear the memory
                        clearCache()

                     
if __name__ == "__main__":
    main()