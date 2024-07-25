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
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K27me3", "H3K36me3", "H3K4me1"]
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
    directory = f"{input_directory}/*_train.bed.gz"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    output_file = f"{output_directory}/trainLabels.h5"

    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for file in files:
            if cell in file:
                df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep="\t", header=None)
                # convert to a dictionary
                df_dict = df.to_dict()
                dict[cell] = df.values

    # create an empty hdf5 file
    with h5py.File(output_file, 'w') as f:
        pass

    # append data to hdf5 file as a new dataset
    with h5py.File(output_file, 'a') as f:
        for cell in dict:
           print(cell)
           dict_list = dict[cell]
           # convert everything to string
           dict_list = np.array([[str(j) for j in i] for i in dict_list])
           # change the dtype to h5py string
           dict_list = np.array([[np.string_(j) for j in i] for i in dict_list])
           # flatten the list
           f.create_dataset(cell + "_labels", data=dict_list,compression='gzip', compression_opts=6)

    # print out the dictionary structure
    for cell in dict.keys():
        print(f"\n\tcell: {cell}\n\t\tfiles: {dict[cell].shape}")
    print("============")

    return dict


def compressHoldoutData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    types = ["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K27me3", "H3K36me3", "H3K4me1"]
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
    directory = f"{input_directory}/*Enhancer.bed.gz"
    cellLines = ["A549","HepG2", "K562", "MCF7"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    studies = np.unique([i.split("_")[1] for i in files])
    outfile = f"{output_directory}/holdoutLabels.h5"

    dict = {}
    for cell in cellLines:
        if cell not in dict:
            dict[cell] = {}
        for study in studies:
            if study not in dict[cell]:
                dict[cell][study] = {}
            for file in files:
               if cell in file and study in file:
                   df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep="\t", header=None)
                   dict[cell][study] = df.values        

    # print out the dictionary structure
    for cell in dict.keys():
        for study in dict[cell].keys():
            print(f"\n\tcell: {cell}\n\t\t\tstudy: {study}\n\t\t\t\tfiles: {dict[cell][study].shape}\n\nfirst 5: {dict[cell][study][:5]}")
    print("============")

    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    # append data to hdf5 file as a new dataset
    with h5py.File(outfile, 'a') as f:
        for cell in dict:
           for study in dict[cell]:
               dict_list = dict[cell][study]
               # convert everything to string
               dict_list = np.array([[str(j) for j in i] for i in dict_list])
               # change the dtype to h5py string
               dict_list = np.array([[np.string_(j) for j in i] for i in dict_list])
               # flatten the list
               f.create_dataset(cell + "_" + study + "_labels", data=dict_list,compression='gzip', compression_opts=6)

    # print out current h5 structure
    with h5py.File(outfile, 'r') as f:
        for key in f.keys():
            print(f"key: {key}\tshape: {f[key].shape} \t first 5 elements: {f[key][:5]}")
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
    nested_train = compressTrainData(args.train_input, args.train_output)
    nested_labels = compressTrainLabels(args.train_input, args.train_output)

    # Compressing holdout data and labels
    nested_test = compressHoldoutData(args.holdout_input, args.holdout_output)
    nested_test_labels = compressHoldoutLabels(args.holdout_input, args.holdout_output)

if __name__ == '__main__':
    main()
