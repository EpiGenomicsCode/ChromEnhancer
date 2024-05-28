from util import dataset as DS
from util.utils import *
from torch.utils.data import DataLoader
import glob
import re
# add command line arguments
import argparse
from itertools import combinations

# add command line arguments
parser = argparse.ArgumentParser(description='Run the study')
parser.add_argument('--fileInput', type=str, help='Location of training and test data', default="./data/CHR_NETWORK/")
parser.add_argument('--fileOutput', type=str, help='Location of training and test data', default="./output/")
parser.add_argument('--parameterCHR', action='store_false', help='Run the parameter study with chromosome dropout')
parser.add_argument('--parameterCLD', action='store_false', help='Run the parameter study with cell line dropout')
parser.add_argument('--parameterLDS', action='store_false', help='Run the parameter study with Large Dataset')

# optional
parser.add_argument('--cellLine', nargs='+', help='Run the study on the cellLine', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--chromData', nargs='+', help='Run the study using the following chromatin datasets', default=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"])
parser.add_argument('--chrPair', nargs='+', help='Run the study dropping out chromosome pairs', default=["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"])
parser.add_argument('--index', nargs='+', help='Run the study on the index', default=["-1","-2"])
parser.add_argument('--model', nargs='+', help='Run the study on the model', default=["1", "2", "3", "4", "5", "6"])
parser.add_argument('--batch_size', type=int, help='Run the study on the batch size', default=1024)
parser.add_argument('--bin_size', type=int, help='How many bins to use when loading the data', default=65536)
parser.add_argument('--epochs', type=int, help='Run the study on the epochs', default=20)

args = parser.parse_args()

ALLCELLS = ["A549", "MCF7", "HepG2", "K562"]
ALLCHROM = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"]
#ALLCHROM = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
 
def main():
    clearCache()
    fileInput = args.fileInput
    fileOutput = args.fileOutput
    cellLine = args.cellLine
    chromData = args.chromData
    chrPair = args.chrPair
    index = args.index
    epochs = args.epochs
    batch_size = args.batch_size
    bin_size = args.bin_size

    print("Running arguments: {}".format(args))
    seedEverything()

    if not args.parameterCHR:
        print("Running Parameter Study with Chromosome Dropout")
        parameterCHR(fileInput, fileOutput, cellLine, chrPair, epochs, batch_size, bin_size)

    if not args.parameterCLD:
        print("Running Parameter Study with Cell Line Dropout")
        parameterCLD(fileInput, fileOutput, cellLine, chromData, epochs, batch_size, bin_size)

    if not args.parameterLDS:
        print("Running Parameter Study with Large Dataset")
        parameterLDS(fileInput, fileOutput)

def parameterCHR(fileInput, outputPath, cellLines=["A549", "MCF7", "HepG2", "K562"], chrPairs=["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"], epochs=20, batch_size=1024, bin_size=4096):
    """
        Hyperparameter search of different model architectures
        with chromosome dropout
    """
    fileLocation = fileInput
    fileOutput = outputPath

    #print(cellLines)
    params = []
    for cellLine in cellLines:
        for types in args.index: 
            # go through each study
            for chromPair in chrPairs:
                data1, data2 = chromPair.split("-")
                # process the data for each model train, test and test, train
                for data  in [ [data1, data2], [data2, data1]]:
                    test = data[0]
                    valid = data[1]
                    for modelType in args.model:
                        name = f"study_{chromPair}_test_{test}_valid_{valid}_model{modelType}_clkeep_{cellLine}_chkeep_{'-'.join(ALLCHROM)}_type{types}"
                        params.append( [
                            chromPair, # chr10-chr17
                            test, # chr10
                            valid, # chr17
                            ALLCHROM, # chUse
                            [cellLine], # clUse: ["A549"]
                            [cellLine],
                            types, # -1
                            name, # study_chr10-chr17_test_chr10_valid_chr17_model1_clkeep_A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII_type-1
                            epochs, # 20
                            batch_size, # 2048
                            bin_size,
                            modelType, # 1
                            fileLocation, # ./Data/220802_DATA/ 
                            fileOutput # ./output/
                        ] )
    
    parseParam("paramCHR.log", params)

def parameterCLD(fileInput, outputPath, cellUse=["A549", "MCF7", "HepG2", "K562"], chromUse=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"], epochs=20, batch_size=1024, bin_size=4096):
    """
        Hyperparameter search of different model architectures
        combining cell lines to make cell-type invariant model
    """
    cellHoldout = list(set(ALLCELLS) - set(cellUse))
    fileLocation = fileInput
    fileOutput = outputPath

    params = []
    for types in args.index:
       for modelType in args.model:
           name = f"CLD_study_-_test_-_valid_-_model{modelType}_clkeep_{'-'.join(cellUse)}_chkeep_{'-'.join(chromUse)}_type{types}"
           params.append(["", "", "", chromUse, cellUse, cellHoldout, types, name, epochs, batch_size, bin_size, modelType, fileLocation, fileOutput])

    # run the study
    parseParam("paramCLD.log", params)

def parameterLDS(fileInput, outputPath):
    fileLocation = fileInput
    fileOutput = outputPath

    chromPair = "chr12-chr8"
    cellLine = "K562"

    for modelType in args.model:
        name = f"study_{chromPair}_test_chr12_valid_chr8_model{modelType}_clkeep_{cellLine}"
        param1 = ["chr12-chr8", "chr12", "chr8", [], [cellLine], [cellLine], "", f"LargeDataset1_{modelType}", args.epochs, args.batch_size, args.bin_size, modelType, fileLocation, fileOutput]
        parseParam("paramLDS.log", [param1])
#        param2 = ["chr12-chr8", "chr8", "chr12", [], [cellLine], [cellLine], "", f"LargeDataset2_{modelType}", args.epochs, args.batch_size, args.bin_size, modelType, fileLocation, fileOutput]
#        parseParam("paramLDS.log", [param2])

def simulation_started(started_file, simulation_name):
    sim =  open(started_file, "r")
    for line in sim:
        print(line)
        if simulation_name in line:
            return True
    return False

# Function to mark a simulation as started
def mark_simulation_as_started(started_file,simulation_name):
    with open(started_file, "a") as f:
        f.write(simulation_name + "\n")

def parseParam(startedFile, params):
    params.sort(key=lambda x: x[6])
    # create a new file if it does not exist
    if not os.path.exists(startedFile):
        with open(startedFile, "w") as f:
            f.write("")

    ## print the params
    #params.append(["", "", "",chromUse, cellUse, cellHoldout, types, name, epochs, batch_size, bin_size, modelType, fileLocation])

    for i in params:
        print(i[7])

    for i in tqdm.tqdm(params):
        study = i[0]
        test = i[1]
        valid = i[2]
        chrUse = i[3]
        cellUse = i[4]
        cellHold = i[5]
        types = i[6]
        name = i[7]
        epochs = i[8]
        batch_size = i[9]
        bin_size = i[10]
        modelconfig = i[11]
        fileLocation = i[12]
        fileOutput = i[13]
        clearCache()
        if simulation_started(startedFile, name):
            print(f"{name} has already been started, skipping.")
            continue
        else:
            mark_simulation_as_started(startedFile, name)
            runStudy(study, test, valid, chrUse, cellUse, cellHold, types, name, epochs, batch_size, bin_size, modelconfig, fileLocation, fileOutput)

def runStudy(study, test, valid, chrUse, cellUse, cellHold, types, name, epochs, batch_size, bin_size, modelconfig, fileLocation, fileOutput):
    clearCache()
    ds_train, ds_test, ds_valid = DS.getData(   trainLabel=study,
                                                testLabel=test,
                                                validLabel=valid,
                                                chrUse=chrUse,
                                                cellUse=cellUse,
                                                cellHold=cellHold,
                                                bin_size=bin_size,
                                                fileLocation=fileLocation,
                                                dataTypes =types)
    # cast each dataset to a pytorch dataloader
    train_loader = DataLoader(ds_train, batch_size=batch_size )
    test_loader = DataLoader(ds_test, batch_size=batch_size )
    valid_loader = DataLoader(ds_valid, batch_size=batch_size )

    # load the model
    if "LARGE" in fileLocation:
        model = loadModel(modelconfig, name, 33000)
    else:
        model = loadModel(modelconfig, name, 800)
    # run the model
    model = trainModel(model, train_loader, test_loader, valid_loader, epochs, fileOutput)

    del ds_train, ds_test, ds_valid, train_loader, test_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
