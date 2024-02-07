import pandas as pd
import glob
import os
import numpy as np
import h5py
import pandas as pd
import os
import gc 
import multiprocessing 
import glob
import h5py
import numpy as np
import pandas as pd




def compressTrainData(directory="./TRAIN/*", output="./TRAIN/trainData.h5", chunk_size=10000):
    """
    Compresses training data into an HDF5 file with gzip compression.

    Args:
        directory (str): Path to the directory containing the input training data files.
        output (str): Path to the output HDF5 file.
        chunk_size (int): Number of samples to process in each chunk.

    Returns:
        None
    """
    print(f"directory: {directory}, output: {output}")

    # Get a list of all the input training data files
    files = sorted(glob.glob(directory + "*track"))

    # Remove ".label" files
    files = sorted([i.split("/")[-1] for i in files])

    # Get unique cell lines and chromosome types from the file names
    cellLine = np.unique([i.split("_")[0] for i in files])
    chrtypes = np.unique([i.split("_")[1] for i in files])


    # get the length of one file
    datashape = pd.read_csv(directory[:-1]+files[0], sep=" ", header=None).shape
    length = datashape[0]
    columns = datashape[1]*len(files)
    print(datashape)

    for cell in cellLine:
        for chrt in chrtypes:
            print(f"\tProcessing {cell} {chrt}  from {directory} to {output}\n")

            # Create an empty dataset 
            with h5py.File(output, 'a') as f:
                pass


            # get the number of chunks
            num_chunks = length // chunk_size
            if length % chunk_size != 0:
                num_chunks += 1

            # Iterate over the chunks
            for i in range(num_chunks):
                data = []
                # go through every file
                for f in files:
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





    
def compressTrainLables(directory =  "./TRAIN/*label*",outfile="./TRAIN/trainLabels.h5"):
    files = glob.glob(directory)
    files = [i.split("/")[-1] for i in files]

    #  create an empty hdf5 file
    with h5py.File(outfile, 'w') as f:
        pass

    for file in files:
        print(f"\tProcessing {file}  from {directory} to {outfile}\n")
        df = pd.read_csv(directory[:directory.rindex("/")+1]+file, sep=" ", header=None)
        with h5py.File(outfile, 'a') as f:
            f.create_dataset(file.split(".")[0], data=df.values,compression='gzip', compression_opts=6)

            

def compressHoldoutLabels(outfile="./HOLDOUT/testLabels.h5"):
    compressTrainLables("./HOLDOUT/*label*", outfile)

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


    # nested_test_labels = compressHoldoutLabels()
    # nested_labels = compressTrainLables()

    # nested_test = compressHoldoutData()
    # nested_train = compressTrainData()


    
if __name__ == '__main__':
    main()