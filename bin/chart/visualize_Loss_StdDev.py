import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_data(folder_path, modelType, output_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    # Get a list of CSV files in the folder
    files = [file for file in os.listdir(folder_path) if file.endswith(".csv") and modelType in file]
    num_files = len(files)
    if num_files == 0:
        print("No CSV files found in the folder.")
        return
    
    # Load CSV files into pandas DataFrame
    dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in files]
    
    # Calculate the mean and standard error
    means = np.mean([df.iloc[:, 1] for df in dfs], axis=0)
    std_errs = np.std([df.iloc[:, 1] for df in dfs], axis=0) / np.sqrt(num_files)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(means, label='Mean')
    plt.fill_between(range(len(means)), means - std_errs, means + std_errs, color='skyblue', alpha=0.3, label='Standard Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average of Loss with Standard Error')
    plt.legend()
    plt.xticks(range(len(means)))  # Assuming x-axis ticks correspond to index from 0 to 19
    plt.xlim(0, 19)  # Limit x-axis from 0 to 19
    plt.savefig(output_path, format='svg', dpi=300)  # Save plot to output SVG file with DPI of 300

if __name__ == "__main__":
    # Check if folder path and output path are provided as command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <folder_path> <output_path>")
    else:
        folder_path = sys.argv[1]
        modelType = sys.argv[2]
        output_path = sys.argv[3]
        plot_data(folder_path, modelType, output_path)
