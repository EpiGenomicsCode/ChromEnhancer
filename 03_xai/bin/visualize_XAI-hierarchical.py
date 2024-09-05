import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate heatmaps from CSV files and save as SVG')
    parser.add_argument('--clustData', type=str, required=True, help='Path to the data to be clustered in TSV format')
    parser.add_argument('--linkData', type=str, required=True, help='Path to the data that will be row-linked to the cluster data in TSV format')
    parser.add_argument('--linkVMin', type=int, default=-1, help='VMin for linked data heatmap')
    parser.add_argument('--linkVMax', type=int, default=1, help='VMin for linked data heatmap')
    parser.add_argument('--outputFile', type=str,  default="output.svg", help='Output SVG file')
    parser.add_argument('--outputDendrogram', action='store_false', help='Output dendrogram associated with clustering')
    parser.add_argument('--outputDendrogramFile', type=str,  default="dendrogram.svg", help='Output dendrogram SVG file')
    return parser.parse_args()

# Function to bin data in 100 increments and sum
def bin_and_sum(df, bin_size=100):
    n_rows, n_cols = df.shape
    # Determine the number of bins
    n_bins = n_cols // bin_size
    
    # Initialize a list to store binned data
    binned_data = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        bin_sum = df.iloc[:, start:end].sum(axis=1)
        binned_data.append(bin_sum)
    
    # Convert the list of Series to a DataFrame
    binned_df = pd.DataFrame(binned_data).T
    return binned_df

def hierarchical_clustering(df):
    linked = linkage(df, method='ward')  # you can choose other methods like 'single', 'complete', etc.
    return linked

def reorder_dataframe(df, order):
    return df.iloc[order]

def plot_dendrogram(linked, output_denfile):
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.savefig(output_denfile, format='svg')

# Main function to generate heatmaps and save as SVG
def generate_heatmaps(file1, file2, linkvmin, linkvmax, output_svg, output_den, output_denfile):
    print("Loading data...")
    # Read CSV files into DataFrames
    df1 = pd.read_csv(file1, sep="\t")
    df2 = pd.read_csv(file2, sep="\t")

    # Bin and sum the data in 100-bin increments
    binned_df1 = bin_and_sum(df1)
    binned_df2 = bin_and_sum(df2)
    print("Data loaded")

    print("Clustering...")
    # Perform clustering and get the order
    linked = hierarchical_clustering(binned_df1)

    if not output_den:
        plot_dendrogram(linked, output_denfile)

    # Get the order from the dendrogram
    dendro = dendrogram(linked, no_plot=True)
    order = dendro['leaves']

    # Reorder the dataframes
    df1_reordered = reorder_dataframe(binned_df1, order)
    df2_reordered = reorder_dataframe(binned_df2, order)

    print("Clustering complete")

    # Create subplots for side-by-side heatmaps
    fig, axes = plt.subplots(1, 2)

    # Generate heatmaps for each file
    heatmap1 = axes[0].imshow(df1_reordered, cmap='PRGn', aspect='auto', interpolation='none', vmax=8)
    axes[0].set_title('Cluster Data')

    heatmap2 = axes[1].imshow(df2_reordered, cmap='seismic', aspect='auto', interpolation='none', vmin=linkvmin, vmax=linkvmax)
    axes[1].set_title('Linked Data')

    # Add colorbars
    fig.colorbar(heatmap1, ax=axes[0], shrink=0.8)
    fig.colorbar(heatmap2, ax=axes[1], shrink=0.8)

    # Adjust layout
    plt.tight_layout()

    # Save plot as SVG
    plt.savefig(output_svg, format='svg')

    # Display the plot (optional)
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Generate heatmaps and save as SVG
    generate_heatmaps(args.clustData, args.linkData, args.linkVMin, args.linkVMax, args.outputFile, args.outputDendrogram, args.outputDendrogramFile)

