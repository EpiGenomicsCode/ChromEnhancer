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

def compressTrainData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "labels" in i and i.endswith('.chromtrack.gz')]
    output = f"{output_directory}/trainData.h5"
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

def compressTrainLabels(input_directory, output_directory):
    directory = f"{input_directory}/*label*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    outfile = f"{output_directory}/trainLabels.h5"

    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    for file in files:
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(outfile, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)

    with h5py.File(outfile, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape}")
    print("============")

    return dict

def compressHoldoutData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and not "labels" in i and i.endswith('.chromtrack.gz')]
    output = f"{output_directory}/holdoutData.h5"
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

def compressHoldoutLabels(input_directory, output_directory):
    directory = f"{input_directory}/*label*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    outfile = f"{output_directory}/holdoutLabels.h5"

    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    for file in files:
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(outfile, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)

    with h5py.File(outfile, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape}")
    print("============")

    return dict

def main():
    parser = argparse.ArgumentParser(description='Compress data and labels.')
    parser.add_argument('--train_input', type=str, required=True, help='Input directory for training data')
    parser.add_argument('--train_output', type=str, required=True, help='Output directory for training data H5 files')
    parser.add_argument('--holdout_input', type=str, required=True, help='Input directory for holdout data')
    parser.add_argument('--holdout_output', type=str, required=True, help='Output directory for holdout data H5 files')

    args = parser.parse_args()

    # Compressing training data and labels
    #nested_train = compressTrainData(args.train_input, args.train_output)
    #nested_labels = compressTrainLabels(args.train_input, args.train_output)

    # Compressing holdout data and labels
    nested_test = compressHoldoutData(args.holdout_input, args.holdout_output)
    nested_test_labels = compressHoldoutLabels(args.holdout_input, args.holdout_output)

if __name__ == '__main__':
    main()
