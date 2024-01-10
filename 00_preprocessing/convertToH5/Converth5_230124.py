import pandas as pd
import glob
import os
import numpy as np
import h5py
import pandas as pd
import os

import multiprocessing 

def saveDict(data, directory, output):
    #  create an empty hdf5 file 
    with h5py.File(output, 'w') as f:
        pass  # An empty "with" statement body creates an empty file
     
    #  go through dictionary and sort files
    for cell in data:
        for study in data[cell]:
            for file_type in data[cell][study]:
                files = data[cell][study][file_type]
                types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
                sortedData = []
                for t in types:
                    for f in files:
                        if t in f:
                            sortedData.append(f)
                data[cell][study][file_type] = sortedData

                df = []
                for f in sortedData:
                    df.append(pd.read_csv(directory[:-1]+f, sep=" ", header=None))
                df = pd.concat(df, axis=1)
                
                # append data to hdf5 file as a new dataset
                with h5py.File(output, 'a') as f:
                    f.create_dataset(cell + "_" + study + "_" + file_type, data=df.values,compression='gzip', compression_opts=6)
                
                # print out current h5 structure
                with h5py.File(output, 'r') as f:
                    for key in f.keys():
                        print(f"key: {key}\tshape: {f[key].shape}")
                print("============")

                        
    return data

def compressTrainData(directory="./TRAIN/*", output="./TRAIN/trainData.h5"):
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if "seq" in i]
    study = np.unique([i.split("_")[1] for i in files])
    dict = {}

    # create an empty hdf5 file
    with h5py.File(output, 'w') as f:
        pass

    for file in files:
        print(file)
        # read file and add it to the h5
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(output, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)
                        
                    
    # print out current h5 structure
    with h5py.File(output, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape}")
        print("============")

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

    # nested_train = compressTrainData("./TRAIN/trainData.h5")
    # nested_labels = compressTrainLables("./TRAIN/trainLabels.h5")
    # nested_test = compressHoldoutData("./HOLDOUT/testData.h5")
    # nested_test_labels = compressHoldoutLabels("./HOLDOUT/testLabels.h5")


    
if __name__ == '__main__':
    main()