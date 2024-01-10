import pandas as pd
import glob
import os
import numpy as np
import h5py
import pandas as pd
import os

import multiprocessing 

def compressTrainData(directory="./TRAIN/*", output="./TRAIN/trainData.h5"):
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "label" in i]
    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for file in files:
            if cell in file:
                if "-1_combined" in file:
                    file_type = "-1"
                elif "-2_combined" in file:
                    file_type = "-2"
                else:
                    file_type = "unknown"
                if file_type not in dict[cell]:
                    dict[cell][file_type] = []
                dict[cell][file_type].append(file)

    # create an empty hdf5 file
    with h5py.File(output, 'w') as f:
        pass

    # print out the dictionary structure
    for cell in dict.keys():
        for file_type in dict[cell].keys():
            files = dict[cell][file_type]
            sorted = []
            # sort files based on types
            for type in types:
                for file in files:
                    if type in file:
                        sorted.append(file)
            dict[cell][file_type] = sorted
            df = []
            print(f"\tProcessing {cell} {file_type}  from {directory} to {output}\n\t\t{sorted}\n")
            for file in sorted:
                df.append(pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None))
            df = pd.concat(df, axis=1)
            with h5py.File(output, 'a') as f:
                f.create_dataset(cell+"_"+file_type, data=df.values,compression='gzip', compression_opts=6)
    return dict

def compressTrainLables(outfile="./TRAIN/trainLabels.h5"):
    directory =  "./TRAIN/*label*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    
    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    for file in files:
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(outfile, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)

    return dict

def compressHoldoutLabels(outfile="./HOLDOUT/testLabels.h5"):
    directory =  "./HOLDOUT/*label*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    
    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass
   
    for file in files:
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(outfile, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)

    return dict


def compressHoldoutData(output="./HOLDOUT/testData.h5"):
    compressTrainData("./HOLDOUT/*", output)

def main():
    #  run the functions in parallel
    pool = multiprocessing.Pool(processes=4)
    pool.apply_async(compressTrainData)
    pool.apply_async(compressTrainLables)
    pool.apply_async(compressHoldoutData)
    pool.apply_async(compressHoldoutLabels)
    pool.close()
    pool.join()

    # nested_labels = compressTrainLables("./TRAIN/trainLabels.h5")
    # nested_test_labels = compressHoldoutLabels("./HOLDOUT/testLabels.h5")
    # nested_train = compressTrainData("./TRAIN/trainData.h5")
    # nested_test = compressHoldoutData("./HOLDOUT/testData.h5")


    
if __name__ == '__main__':
    main()