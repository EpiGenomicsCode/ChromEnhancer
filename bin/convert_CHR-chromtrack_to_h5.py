import argparse
import pandas as pd
import glob
import os
import numpy as np
import h5py
import pandas as pd
import os

import multiprocessing 

import sys

def logicOrder(directory, cellLines, chromList, files, output="temp.h5"):
    """
    Takes in a list of cell lines, a list of chromosomes, and a list of files and
    returns a nested dictionary with cell lines and chromosomes as keys.
    """
   # Initialize empty nested dictionary
    data = {}

    # Loop over cell lines
    for cell in cellLines:
        # Initialize nested dictionary for cell line
        data[cell] = {}

        # Loop over chromList
        for chrom in chromList:
            # Initialize nested dictionary for chromosome
            data[cell][chrom] = {}

            # Loop over files
            for file in sorted(files):
                # Check if file contains cell line and chromosome
                if cell in file and chrom in file:
                    # Get file type (-1 or -2)
                    if "-1_combined" in file:
                        file_type = "-1"
                        # if key does not exist, create it
                        if "-1" not in data[cell][chrom]:
                            data[cell][chrom]["-1"] = []
                        
                        data[cell][chrom]["-1"].append(file)
                    elif "-2_combined" in file:
                        file_type = "-2"
                        # if key does not exist, create it
                        if "-2" not in data[cell][chrom]:
                            data[cell][chrom]["-2"] = []
                        data[cell][chrom]["-2"].append(file)
                    else:
                        file_type = "unknown"
    # # Useful for debugging structure
    # # -print the dictionary structure
    # for cell in data.keys():
    #     for chrom in data[cell].keys():
    #         for file_type in data[cell][chrom].keys():
    #             print(f"\n\tcell: {cell}\n\t\tchrom: {chrom}\n\t\t\tfile_type: {file_type}\n\t\t\t\tfiles: {data[cell][chrom][file_type]}")
    # print("============")

    data = saveDict(data, directory, output)
    return data

def saveDict(data, directory, output):
    #  create an empty hdf5 file 
    with h5py.File(output, 'w') as f:
        pass  # An empty "with" statement body creates an empty file
     
    #  go through dictionary and sort files
    for cell in data:
        for chrom in data[cell]:
            for file_type in data[cell][chrom]:
                files = data[cell][chrom][file_type]
                types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
                sortedData = []
                for t in types:
                    for f in files:
                        if t in f:
                            sortedData.append(f)
                data[cell][chrom][file_type] = sortedData

                df = []
                for f in sortedData:
                    df.append(pd.read_csv(directory[:-1]+f, sep=" ", header=None))
                df = pd.concat(df, axis=1)
                
                # append data to hdf5 file as a new dataset
                with h5py.File(output, 'a') as f:
                    f.create_dataset(cell + "_" + chrom + "_" + file_type, data=df.values,compression='gzip', compression_opts=6)
                
                # print out current h5 structure
                with h5py.File(output, 'r') as f:
                    for key in f.keys():
                        print(f"key: {key}\tshape: {f[key].shape}")
                print("============")

                        
    return data

def compressTrainData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "labels" in i and i.endswith('.chromtrack.gz')]
    chromList = np.unique([i.split("_")[1] for i in files if i.endswith('.chromtrack.gz')])
    output_file = f"{output_directory}/trainData.h5"
    nested_dict = logicOrder(directory, cellLines, chromList, files, output_file)
    return nested_dict

def compressHoldoutData(input_directory, output_directory):
    directory = f"{input_directory}/*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "labels" in i and i.endswith('.chromtrack.gz')]
    chromList = np.unique([i.split("_")[1] for i in files if i.endswith('.chromtrack.gz')])
    output_file = f"{output_directory}/holdoutData.h5"
    nested_dict = logicOrder(directory, cellLines, chromList, files, output_file)
    return nested_dict

def compressTrainLabels(input_directory, output_directory):
    directory = f"{input_directory}/*bed"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)

    files = [i.split("/")[-1] for i in files]
    chromList = np.unique([i.split("_")[1] for i in files])
    output_file = f"{output_directory}/trainLabels.h5"

    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for chrom in chromList:
            if chrom not in dict[cell]:
                dict[cell][chrom] = {}
            for file in files:
                if cell in file and chrom in file:
                    df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep="\t", header=None)
                    # convert to a dictionary
                    df_dict = df.to_dict()
                    dict[cell][chrom] = df.values

    # create an empty hdf5 file
    with h5py.File(output_file, 'w') as f:
        pass

    # append data to hdf5 file as a new dataset
    with h5py.File(output_file, 'a') as f:
        for cell in dict:
            for chrom in dict[cell]:
                print(cell + "\t" + chrom)
                dict_list = dict[cell][chrom]
                # convert everything to string
                dict_list = np.array([[str(j) for j in i] for i in dict_list])
                # change the dtype to h5py string
                dict_list = np.array([[np.string_(j) for j in i] for i in dict_list])
                # flatten the list
                f.create_dataset(cell + "_" + chrom + "_labels", data=dict_list,compression='gzip', compression_opts=6)

    # print out current h5 structure
    with h5py.File(output_file, 'r') as f:
        for key in f.keys():
            # print key data and shape
            print(f"key: {key}\tshape: {f[key].shape}\n\t\t{f[key][:5]}")
    print("============")


    return dict

def compressHoldoutLabels(input_directory, output_directory):
    directory = f"{input_directory}/*bed"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    studies = np.unique([i.split("_")[1] for i in files])
    chrs = np.unique([i.split("_")[2].split(".")[0] for i in files])
    output_file = f"{output_directory}/holdoutLabels.h5"

    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for chr in chrs:
            if chr not in dict[cell]:
                dict[cell][chr] = {}
            for study in studies:
                if study not in dict[cell][chr]:
                    dict[cell][chr][study] = {}
                for file in files:
                    if cell in file and study in file and chr in file:
                        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep="\t", header=None)
                        dict[cell][chr][study] = df.values                

    # print out the dictionary structure
    for cell in dict.keys():
        for chr in dict[cell].keys():
            for study in dict[cell][chr].keys():
                print(f"\n\tcell: {cell}\n\t\tchr: {chr}\n\t\t\tstudy: {study}\n\t\t\t\tfiles: {dict[cell][chr][study].shape}\n\nfirst 5: {dict[cell][chr][study][:5]}")
    print("============")

    # create an empty hdf5 file
    with h5py.File(output_file, 'w') as f:
        pass

    # append data to hdf5 file as a new dataset
    with h5py.File(output_file, 'a') as f:
        for cell in dict:
            for chr in dict[cell]:
                for study in dict[cell][chr]:
                    dict_list = dict[cell][chr][study]
                    # convert everything to string
                    dict_list = np.array([[str(j) for j in i] for i in dict_list])
                    # change the dtype to h5py string
                    dict_list = np.array([[np.string_(j) for j in i] for i in dict_list])
                    # flatten the list
                    f.create_dataset(cell + "_" + chr + "_" + study + "_labels", data=dict_list,compression='gzip', compression_opts=6)
                    
    # print out current h5 structure
    with h5py.File(output_file, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape} \t first 5 elements: {f[key][:5]}")
    print("============")

    return dict

def main():
    parser = argparse.ArgumentParser(description='Compress data and labels.')
    parser.add_argument('--train_input', 
                        type=str, 
                        help='Input directory for training data',
                        default="CHR-TRAIN")
    parser.add_argument('--train_output', 
                            type=str, 
                            help='Output directory for training data H5 files', 
                            default="CHR-TRAINOUT")
    parser.add_argument('--holdout_input', 
                        type=str, 
                        help='Input directory for holdout data',
                        default="CHR-HOLDOUT")
    parser.add_argument('--holdout_output', 
                            type=str, 
                            help='Output directory for holdout data H5 files',
                            default="CHR-HOLDOUTOUT")


    args = parser.parse_args()
    
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.holdout_output, exist_ok=True)

    # Compressing training data and labels
    #nested_train = compressTrainData(args.train_input, args.train_output)
    #nested_labels = compressTrainLabels(args.train_input, args.train_output)

    # Compressing holdout data and labels
    #nested_test = compressHoldoutData(args.holdout_input, args.holdout_output)
    nested_test_labels = compressHoldoutLabels(args.holdout_input, args.holdout_output)

    # take in the generated label file and print out the score 

    #f = h5py.File(f"{args.train_output}/trainLabels.h5", 'r')
    #for key in f.keys():
    #    print(f"key: {key}\tshape: {f[key].shape}\n\t\t{f[key][:5]}")

        # expected output:
        # key: A549_chr10-chr17_labels    shape: (702713, 6)
                #  [
                # [b'chr8' b'125090960' b'125091960' b'chr8:125091000-125092000' b'1' b'.']
                #  [b'chr2' b'79630080' b'79631080' b'chr2:79630000-79631000' b'1' b'.']
                #  [b'chr1' b'162597000' b'162598000' b'chr1:162597000-162598000' b'0' b'.']
                #  [b'chr16' b'10282000' b'10283000' b'chr16:10282000-10283000' b'0' b'.']
                #  [b'chr4' b'35383000' b'35384000' b'chr4:35383000-35384000' b'0' b'.']
                # ]
if __name__ == '__main__':
    main()
