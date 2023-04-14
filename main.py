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
parser.add_argument('--sequence', action='store_false', help='Run the sequence study')
parser.add_argument('--parameter', action='store_false', help='Run the parameter study')
parser.add_argument('--parameterCLD', action='store_false', help='Run the parameter study with cell line dropout')
parser.add_argument('--parameterCHD', action='store_false', help='Run the parameter study with chromatin droupout')

# optional cellLine
parser.add_argument('--cellLine', nargs='+', help='Run the study on the cellLine', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--index', nargs='+', help='Run the study on the index', default=["-1","-2"])
parser.add_argument('--model', nargs='+', help='Run the study on the model', default=["1", "2", "3", "4", "5"])
parser.add_argument('--batch_size', type=int, help='Run the study on the batch size', default=32)
parser.add_argument('--bin_size', type=int, help='How many bins to use when loading the data', default=2)
parser.add_argument('--epochs', type=int, help='Run the study on the epochs', default=20)

args = parser.parse_args()

def main():
    clearCache()
    cellLine = args.cellLine
    index = args.index
    epochs = args.epochs
    batch_size = args.batch_size
    bin_size = args.bin_size

    print("Running arguments: {}".format(args))
    seedEverything()


    if not args.sequence:
        print("Running Sequence Study")
        sequenceStudy(epochs, batch_size)

    if not args.parameter:
        print("Running Parameter Study")
        paramatersStudy(cellLine, index, epochs, batch_size, bin_size)

    if not args.parameterCLD:
        print("Running Parameter Study with Cell Line Dropout")
        CellLineDropout(cellLine, index, epochs, batch_size, bin_size)

    if not args.parameterCHD:
        print("Running Parameter Study with Chromatin Dropout")
        ChromatineDropout(cellLine, index, epochs, batch_size, bin_size)

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
            name = name + "_model{}".format(i)
            model = loadModel(i, name, input_size=4000)
            print(name)
            print(model)
            model = runHomoModel(model, trainLoader, testLoader, validLoader, epochs)

            # clear the memory
            clearCache()
        
def paramatersStudy(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        only A459 all chromatin types
    """
    cellLines = ["A549", "MCF7", "HepG2", "K562"]
    chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    for id in cellLine:
        for types in ["-1", "-2"]: 
            # go through each study
            for study in studys:
                data1, data2 = study.split("-")
                # process the data for each model train, test and test, train
                for data  in [ [data1, data2], [data2, data1]]:
                    test = data[0]
                    valid = data[1]
                    
                    cellLineDrop = [i for i in cellLines if i != id]
                    ds_train, ds_test, ds_valid = DS.getData(trainLabel=study,
                                                                testLabel=test,
                                                                validLabel=valid,
                                                                chrDrop=[],
                                                                cellLineDrop=cellLineDrop,
                                                                bin_size=bin_size,
                                                                fileLocation="./Data/220802_DATA/", 
                                                                dataTypes =types)
                    
                    # convert to dataloader
                    ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                    ds_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                    ds_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)
                        
                    for modelType in args.model[::-1]:
                        # drop all celllines except the one we are using
                        name = f"Param_study_{id}_types{types}_train_{study}_test_{test}_valid_{valid}_model{modelType}_drop_{'-'.join(cellLineDrop)}"
                        print(name)
                       
                        model = loadModel(modelType, name)
                        print(model)

                        model = runHomoModel(model, ds_train, ds_test, ds_valid, epochs)
                        
                        # clear the memory
                        clearCache()
                        
  
                        


def CellLineDropout(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        all cellLines except A549, all chromatin types
    """
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    for types in ["-1", "-2"]:
        for study in studys:
            # go through each study
            data1, data2 = study.split("-")
            # process the data for each model train, test and test, train
            for data  in [ [data1, data2], [data2, data1]]:
                test = data[0]
                valid = data[1]

                # drop each cellLine type
                for useCells in combinations(cellLine, len(cellLine)-1):
                    useCells = list(useCells)
                    drop = [i for i in cellLine if i not in useCells]
                    name = f"CLD_Data_{'-'.join(useCells)}_test_{test}_valid_{valid}_study_{study}_drop_{'-'.join(drop)}_chromtypes_{'-'.join(chromtypes)}_type_{types}"
                    print(name)
                    ds_train, ds_test, ds_valid = DS.getData(   trainLabel=study,
                                                                testLabel=test,
                                                                validLabel=valid,
                                                                chrDrop=[],
                                                                cellLineDrop=drop,
                                                                bin_size=bin_size,
                                                                fileLocation="./Data/220802_DATA/", 
                                                                dataTypes =types)
                    # cast each dataset to a pytorch dataloader
                    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

                    #  We are only testing on model 4
                    model = loadModel(4, name)
                    print(model)
                    model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)

                    # clear the memory
                    clearCache()
                    

def ChromatineDropout(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        all celllines, all chromatin types except CTCF
    """    
    chromtypes = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]

    for types in ["-1", "-2"]:
        for study in studys:
            # go through each study
            data1, data2 = study.split("-")
            # process the data for each model train, test and test, train
            for data  in [ [data1, data2], [data2, data1]]:
                test = data[0]
                valid = data[1]

                # drop each cellLine type
                for useChrome in combinations(chromtypes, len(chromtypes)-1):
                    useChrome = list(useChrome)
                    #  get the missing chromtypes
                    drop = [i for i in chromtypes if i not in useChrome]
                    name = f"CHD_Data_{'-'.join(useChrome)}_test_{test}_valid_{valid}_study_{study}_drop_{'-'.join(drop)}_type_{types}"
                    print(name)

                    ds_train, ds_test, ds_valid = DS.getData(   trainLabel=study,
                                                                testLabel=test,
                                                                validLabel=valid,
                                                                chrDrop=drop,
                                                                cellLineDrop=[],
                                                                bin_size=bin_size,
                                                                fileLocation="./Data/220802_DATA/", 
                                                                dataTypes =types)
                    
                    # cast each dataset to a pytorch dataloader
                    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
                    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

                    #  We are only testing on model 4
                    model = loadModel(4, name)
                    print(model)
                    model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)
                    clearCache()
                    
          

if __name__ == "__main__":
    main()