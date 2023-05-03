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
parser.add_argument('--parameterLDS', action='store_false', help='Run the parameter study with Large Dataset')

# optional cellLine
parser.add_argument('--cellLine', nargs='+', help='Run the study on the cellLine', default=["A549", "MCF7", "HepG2", "K562"])
parser.add_argument('--index', nargs='+', help='Run the study on the index', default=["-1","-2"])
parser.add_argument('--model', nargs='+', help='Run the study on the model', default=["1", "2", "3", "4", "5"])
parser.add_argument('--batch_size', type=int, help='Run the study on the batch size', default=2048)
parser.add_argument('--bin_size', type=int, help='How many bins to use when loading the data', default=65536)
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

    if not args.parameterLDS:
        print("Running Parameter Study with Large Dataset")
        LargeDataset()


def sequenceStudy(epochs=20, batch_size=64):
    cellLines = ["A549", "MCF7", "HepG2", "K562"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]
    chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]

    fileLocation = "./Data/230124_CHR-Data_Sequence/"
    params = []
    for cellLine in cellLines:
        for study in studys:
            test, valid = study.split("-")
            for t,v in [[test, valid], [valid, test]]:
                for modelType in args.model:
                    name = f"sequence_study_{study}_test_{t}_valid_{v}_model{modelType}_clkeep_{cellLine}_type-1"
                    # study, test, valid, chrUse, clUse, types, name, epochs, batch_size,modelconfig, fileLocation, bin_size=1024)
                    params.append([study, t, v, chromatine, [cellLine], "", name, epochs, batch_size, modelType, fileLocation])

    parseParam("sequence.log", params)

def paramatersStudy(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        only A459 all chromatin types
    """
    cellLines = ["A549", "MCF7", "HepG2", "K562"]
    chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]
    fileLocation = "./Data/220802_DATA/"
    
    params = []
    for cellLine in cellLines:
        for types in ["-1", "-2"]: 
            # go through each study
            for study in studys:
                data1, data2 = study.split("-")
                # process the data for each model train, test and test, train
                for data  in [ [data1, data2], [data2, data1]]:
                    test = data[0]
                    valid = data[1]
                    for modelType in args.model:
                        name = f"study_{study}_test_{test}_valid_{valid}_model{modelType}_clkeep_{cellLine}_chkeep_{'-'.join(chromatine)}_type{types}"
                        params.append( [
                            study, # chr10-chr17
                            test, # chr10
                            valid, # chr17
                            chromatine, # chUse
                            [cellLine], # clUse: ["A549"]
                            types, # -1
                            name, # study_chr10-chr17_test_chr10_valid_chr17_model1_clkeep_A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII_type-1
                            epochs, # 20
                            batch_size, # 2048  
                            modelType, # 1
                            fileLocation # ./Data/220802_DATA/                           
                        ] )
    
    parseParam("param.log", params)
                        
def CellLineDropout(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        all cellLines except A549, all chromatin types
    """

    cellLines = ["A549", "MCF7", "HepG2", "K562"]
    chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]
    fileLocation = "./Data/220803_CelllineDATA/"
    params = []
    for types in ["-1", "-2"]:
        for useCells in combinations(cellLine, len(cellLine)-1):
            for modelType in args.model:
                name = f"CLD_model{modelType}_clkeep_{'-'.join(useCells)}_chkeep_{'-'.join(chromatine)}_type{types}"
                params.append(["", "", "",chromatine, list(useCells), types, name, epochs, batch_size, modelType, fileLocation])
                    

    # run the study
    parseParam("CLD.log", params)
                    
def ChromatineDropout(cellLine, index, epochs=3, batch_size=64, bin_size=1024):
    """
        Generates the parameters for the study and runs the study
        all celllines, all chromatin types except CTCF
    """    
    cellLines = ["A549", "MCF7", "HepG2", "K562"]
    chromatine =  ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    studys = ["chr10-chr17", "chr11-chr7", "chr12-chr8", "chr13-chr9", "chr15-chr16"]
    fileLocation = "./Data/220802_DATA/"
    params = [] 
    for types in ["-1", "-2"]:
        for study in studys:
            # go through each study
            data1, data2 = study.split("-")
            # process the data for each model train, test and test, train
            for data  in [ [data1, data2], [data2, data1]]:
                test = data[0]
                valid = data[1]

                # drop each cellLine type
                for useChrome in combinations(chromatine, len(chromatine)-1):
                    useChrome = list(useChrome)
                    for modelType in args.model:
                        name = f"CHD_study_{study}_test_{test}_valid_{valid}_model{modelType}_clkeep_{'-'.join(cellLine)}_chkeep_{'-'.join(useChrome)}_type{types}"
                        params.append([study, test, valid,useChrome, cellLine, types, name, epochs, batch_size, modelType, fileLocation])

    parseParam("CHD.log", params)

def LargeDataset():
    #  hard coded for now
    for model in args.model:
        param1 = ["chr12-chr8", "chr12", "chr8", [], ["K562"], "", f"LargeDataset1_{model}", args.epochs, args.batch_size, model, "./Data/230415_LargeData/", args.bin_size]
        parseParam("LDS.log", [param1])
        param1 = ["chr12-chr8", "chr8", "chr12", [], ["K562"], "", f"LargeDataset2_{model}", args.epochs, args.batch_size, model, "./Data/230415_LargeData/", args.bin_size]
        parseParam("LDS.log", [param1])
        
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
    
    # print the params  
    for i in params:
        print(i[6])
    print(f"len(params): {len(params)}")
    # create a new file if it does not exist
    if not os.path.exists(startedFile):
        with open(startedFile, "w") as f:
            f.write("")

    for i in tqdm.tqdm(params):
        name = i[6]
        if simulation_started(startedFile, name):
            print(f"{name} has already been started, skipping.")
            continue
        else:
            mark_simulation_as_started(startedFile, name)
            runStudy(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], args.bin_size)
        
def runStudy(study, test, valid, chrUse, clUse, types, name, epochs, batch_size,modelconfig, fileLocation, bin_size=1024):
  
    ds_train, ds_test, ds_valid = DS.getData(   trainLabel=study,
                                                testLabel=test,
                                                validLabel=valid,
                                                dataTypes=types,
                                                fileLocation=fileLocation,
                                                chrUse=chrUse,
                                                cellLineUse=clUse,
                                                bin_size=bin_size
                                            )
    
    # cast each dataset to a pytorch dataloader
    train_loader = DataLoader(ds_train, batch_size=batch_size )
    test_loader = DataLoader(ds_test, batch_size=batch_size )
    valid_loader = DataLoader(ds_valid, batch_size=batch_size )

    # load the model
    if "Sequence" in fileLocation:
        model = loadModel(modelconfig, name, 4000)
    elif "Large" in fileLocation:
        model = loadModel(modelconfig, name, 32900)
    else:
        model = loadModel(modelconfig, name)
    # run the model
    model = runHomoModel(model, train_loader, test_loader, valid_loader, epochs)

    del ds_train, ds_test, ds_valid, train_loader, test_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

    

if __name__ == "__main__":
    main()