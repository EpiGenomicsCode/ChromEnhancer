import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to read data from a file
def read_data(file_path):
    # Assuming the file is a CSV with no header and one column
    data = pd.read_csv(file_path, header=None)
    return data[0].values

# Function to plot overlapping histograms and save as PNG
def plot_histograms(data1, data2, bins, range, output_file):
    plt.hist(data1, bins=bins, range=range, alpha=0.5, label='Swarm +')
    plt.hist(data2, bins=bins, range=range, alpha=0.5, label='Swarm -')
    plt.ylim(0, 8000)
    plt.legend(loc='upper right')
    # Adjust layout
    plt.tight_layout()

    # Save plot as SVG
    plt.savefig(output_file, format='svg')
    
    plt.close()

# Main function
def main(file1, file2, output_file):
    data1 = read_data(file1)
    data2 = read_data(file2)
    
    bins = np.arange(-1, 1 + 0.05, 0.05)  # Creating bins from -1 to 1 with a bin width of 0.05
    plot_histograms(data1, data2, bins, (-1, 1), output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot overlapping histograms from two data files.')
    parser.add_argument('file1', type=str, help='Path to the first data file')
    parser.add_argument('file2', type=str, help='Path to the second data file')
    parser.add_argument('output_file', type=str, help='Path to the output PNG file')

    args = parser.parse_args()
    main(args.file1, args.file2, args.output_file)

