import pandas as pd
import glob
import os
import numpy as np
import h5py
import pandas as pd
import os

import multiprocessing 



def logicOrder(directory, cellLines, studies, files, output="temp.h5"):
    """
    Takes in a list of cell lines, a list of studies, and a list of files and
    returns a nested dictionary with cell lines and studies as keys.
    """
   # Initialize empty nested dictionary
    data = {}

    # Loop over cell lines
    for cell in cellLines:
        # Initialize nested dictionary for cell line
        data[cell] = {}

        # Loop over studies
        for study in studies:
            # Initialize nested dictionary for study
            data[cell][study] = {}

            # Loop over files
            for file in sorted(files):
                # Check if file contains cell line and study
                if cell in file and study in file:
                    # Get file type (-1 or -2)
                    if "-1_combined" in file:
                        file_type = "-1"
                        # if key does not exist, create it
                        if "-1" not in data[cell][study]:
                            data[cell][study]["-1"] = []
                        
                        data[cell][study]["-1"].append(file)
                    elif "-2_combined" in file:
                        file_type = "-2"
                        # if key does not exist, create it
                        if "-2" not in data[cell][study]:
                            data[cell][study]["-2"] = []
                        data[cell][study]["-2"].append(file)
                    else:
                        file_type = "unknown"

    #  print the dictionary structure
    for cell in data.keys():
        for study in data[cell].keys():
            for file_type in data[cell][study].keys():
                print(f"\n\tcell: {cell}\n\t\tstudy: {study}\n\t\t\tfile_type: {file_type}\n\t\t\t\tfiles: {data[cell][study][file_type]}")
    print("============")
    data = saveDict(data, directory, output)
    return data

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

def compressTrainData(output="./TRAIN/trainData.h5"):
    directory =  "./TRAIN/*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "labels" in i]
    studies = np.unique([i.split("_")[1] for i in files])
    nested_dict = logicOrder(directory, cellLines, studies, files, output)
    return nested_dict

def compressTrainLables(outfile="./TRAIN/trainLabels.h5"):
    directory =  "./TRAIN/*label*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    studies = np.unique([i.split("_")[1] for i in files])
    
    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for study in studies:
            if study not in dict[cell]:
                dict[cell][study] = {}
            for file in files:
                if cell in file and study in file:
                    df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
                    dict[cell][study] = df.values

    # create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    # append data to hdf5 file as a new dataset
    with h5py.File(outfile, 'a') as f:
        for cell in dict:
            for study in dict[cell]:
                f.create_dataset(cell + "_" + study + "_labels", data=dict[cell][study],compression='gzip', compression_opts=6)

    # print out current h5 structure
    with h5py.File(outfile, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape}")
    print("============")


    return dict

def compressHoldoutLabels(outfile="./HOLDOUT/testLabels.h5"):
    # directory =  "./HOLDOUT/*label*"
    # cellLines = ["A549","HepG2", "K562", "MCF7"]
    # files = glob.glob(directory)
    # files = [i.split("/")[-1] for i in files]
    # studies = np.unique([i.split("_")[1] for i in files])
    # chrs = np.unique([i.split("_")[2].split(".")[0] for i in files])

    # dict = {}
    # for cell in cellLines:
    #     if cell not in dict:
    #         dict[cell] = {}
    #     for chr in chrs:
    #         if chr not in dict[cell]:
    #             dict[cell][chr] = {}
    #         for study in studies:
    #             if study not in dict[cell][chr]:
    #                 dict[cell][chr][study] = {}
    #             for file in files:
    #                 if cell in file and study in file and chr in file:
    #                     df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
    #                     dict[cell][chr][study] = df.values                

    # # print out the dictionary structure
    # for cell in dict.keys():
    #     for chr in dict[cell].keys():
    #         for study in dict[cell][chr].keys():
    #             print(f"\n\tcell: {cell}\n\t\tchr: {chr}\n\t\t\tstudy: {study}\n\t\t\t\tfiles: {dict[cell][chr][study].shape}")
    # print("============")

    # # create an empty hdf5 file
    # with h5py.File(outfile, 'w') as f:
    #     pass

    # # append data to hdf5 file as a new dataset
    # with h5py.File(outfile, 'a') as f:
    #     for cell in dict:
    #         for chr in dict[cell]:
    #             for study in dict[cell][chr]:
    #                 f.create_dataset(cell + "_" + chr + "_" + study + "_labels", data=dict[cell][chr][study],compression='gzip', compression_opts=6)
                    
    # print out current h5 structure
    with h5py.File(outfile, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape}")
    print("============")

    return dict

def compressHoldoutData(output="./HOLDOUT/testData.h5"):
    directory =  "./HOLDOUT/*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i]
    studies = np.unique([i.split("_")[1] for i in files])
    nested_dict = logicOrder(directory, cellLines, studies, files, output)
    return nested_dict

def main():
    # #  run the functions in parallel
    # pool = multiprocessing.Pool(processes=4)
    # pool.apply_async(compressTrainData)
    # pool.apply_async(compressTrainLables)
    # pool.apply_async(compressHoldoutData)
    # pool.apply_async(compressHoldoutLabels)
    # pool.close()
    # pool.join()

    # nested_train = compressTrainData()
    # nested_labels = compressTrainLables("./TRAIN/trainLabels.h5")
    # nested_test = compressHoldoutData("./HOLDOUT/testData.h5")
    nested_test_labels = compressHoldoutLabels("./HOLDOUT/testLabels.h5")


    
if __name__ == '__main__':
    main()