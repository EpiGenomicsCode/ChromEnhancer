import argparse
import pandas as pd
import glob
import numpy as np
import h5py
import os

import gc 
import multiprocessing 

def logicOrder(directory, cellLines, chromList, files, output="temp.h5"):
    """
    Compresses training data into an HDF5 file with gzip compression.
    """
    print(f"directory: {directory}, output: {output}")

    # Rows to process at once
    chunk_size=10000

    # create an empty hdf5 file
    with h5py.File(output, 'w') as f:
        pass

    for cell in cellLines:
        for chrt in chromList:
            print(f"\tProcessing {cell} {chrt}  from {directory} to {output}\n")
            # get the length of one file
            fileSubset = [file for file in files if chrt in file]
#            print(fileSubset)
            datashape = pd.read_csv(directory[:-1]+fileSubset[0], sep=" ", header=None).shape
            length = datashape[0]
            columns = datashape[1]*len(fileSubset)
#            print(datashape)
            print("Final matrix size\nLength: " + str(length) + " Columns: " + str(columns))

            # get the number of chunks
            num_chunks = length // chunk_size
            if length % chunk_size != 0:
                num_chunks += 1

            # Iterate over the chunks
            for i in range(num_chunks):
                data = []
                # go through every file
                for f in fileSubset:
                   # calculate the start and end row for the current chunk
                   start_row = i * chunk_size
                   end_row = min((i + 1) * chunk_size, length)

                   # skip rows and load in the chunk
                   df = pd.read_csv(directory[:-1] + f, sep=" ", header=None, skiprows=start_row, nrows=end_row - start_row)

                   # add it to the data list
                   data.append(df.values)

                # concatenate the data list
                data = np.concatenate(data, axis=1)

                # create the dataset if it does not exist, else append to the end of the dataset
                with h5py.File(output, 'a') as f:
                    if cell+"_"+chrt not in f.keys():
                        print("\t\t\tcreating dataset")
                        f.create_dataset(cell+"_"+chrt, data=data ,compression='gzip', compression_opts=6, maxshape=(length, columns))
                    else:
                        # concatenate the data to the end of the dataset
                        current_length = f[cell+"_"+chrt].shape[0]
                        print(f"\t\t\tappending to dataset left: {length - current_length}")
                        f[cell+"_"+chrt].resize((current_length + data.shape[0], data.shape[1]))      
                        f[cell+"_"+chrt][current_length:] = data

def compressTrainData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["K562"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and i.endswith('.chromtrack.gz')]
    chromList = np.unique([i.split("_")[1] for i in files if i.endswith('.chromtrack.gz')])
    output_file = f"{output_directory}/trainData.h5"
    nested_dict = logicOrder(directory, cellLines, chromList, files, output_file)
    return nested_dict

def compressHoldoutData(input_directory, output_directory):
    directory = f"{input_directory}*"
    cellLines = ["K562"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files if not ".bed" in i and i.endswith('.chromtrack.gz')]
    chromList = np.unique([i.split("_")[1] for i in files if i.endswith('.chromtrack.gz')])
    output_file = f"{output_directory}/holdoutData.h5"
    nested_dict = logicOrder(directory, cellLines, chromList, files, output_file)
    return nested_dict

def compressTrainLabels(input_directory, output_directory):
    directory = f"{input_directory}/*bed"
    cellLines = ["K562"]
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
    cellLines = ["K562"]
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]
    studies = np.unique([i.split("_")[1] for i in files])
    chrs = np.unique([i.split("_")[2].split(".")[0] for i in files])
    output_file = f"{output_directory}/holdoutLabels.h5"

    print(studies)

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

def main():
    parser = argparse.ArgumentParser(description='Compress data and labels.')
    parser.add_argument('--train_input', type=str, help='Input directory for training data', default="CHR-TRAIN")
    parser.add_argument('--train_output', type=str, help='Output directory for training data H5 files', default="CHR-TRAINOUT")
    parser.add_argument('--holdout_input', type=str, help='Input directory for holdout data', default="CHR-HOLDOUT")
    parser.add_argument('--holdout_output', type=str, help='Output directory for holdout data H5 files', default="CHR-HOLDOUTOUT")
    args = parser.parse_args()

    # Compressing training data and labels
#    nested_train = compressTrainData(args.train_input, args.train_output)
#    nested_labels = compressTrainLabels(args.train_input, args.train_output)

    # Compressing holdout data and labels
    nested_test = compressHoldoutData(args.holdout_input, args.holdout_output)
    nested_test_labels = compressHoldoutLabels(args.holdout_input, args.holdout_output)

if __name__ == '__main__':
    main()
