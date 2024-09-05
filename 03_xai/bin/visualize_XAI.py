import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate heatmaps from CSV files and save as SVG')
    parser.add_argument('file1', type=str, help='Path to the first CSV file')
    parser.add_argument('file2', type=str, help='Path to the second CSV file')
    parser.add_argument('file3', type=str, help='Path to the third CSV file')
    parser.add_argument('output', type=str, help='Output SVG file path')
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

# Main function to generate heatmaps and save as SVG
def generate_heatmaps(file1, file2, file3, output_svg):
    print("Loading data...")
    # Read CSV files into DataFrames
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    df3 = pd.read_csv(file3, sep='\t')

    # Bin and sum the data in 100-bin increments
    binned_df1 = bin_and_sum(df1)
    binned_df2 = bin_and_sum(df2)
    binned_df3 = bin_and_sum(df3)
    print("Data loaded")

    print("Clustering...")
    # Hierarchical clustering on the first DataFrame
    clustering = AgglomerativeClustering(n_clusters=8).fit(binned_df1)
    # Get the cluster labels and sort indices
    binned_df1['cluster'] = clustering.labels_
    sorted_df1 = binned_df1.sort_values(by='cluster').reset_index(drop=True)
    sorted_indices = sorted_df1.index
    print(sorted_indices)

    # Apply the sorted indices to DataFrame 2 and DataFrame 3
    sorted_df2 = binned_df2.iloc[sorted_indices].reset_index(drop=True)
    sorted_df3 = binned_df3.iloc[sorted_indices].reset_index(drop=True)

    # Drop the 'cluster' column from the sorted first DataFrame
    sorted_df1 = sorted_df1.drop(columns=['cluster'])

    # Display the sorted DataFrames
    print("Sorted DataFrame 1:\n", sorted_df1)
    print("Orig DataFrame 1:\n", binned_df1)

    print("Sorted DataFrame 2:\n", sorted_df2)
    print("Orig DataFrame 2:\n", binned_df2)

    print("Sorted DataFrame 3:\n", sorted_df3)
    print("Orig DataFrame 3:\n", binned_df3)

    print("Clustering complete")

    # Create subplots for side-by-side heatmaps
    fig, axes = plt.subplots(1, 3)

    # Generate heatmaps for each file
    heatmap1 = axes[0].imshow(sorted_df1, cmap='seismic', aspect='auto', interpolation='none', vmax='20')
    axes[0].set_title('Chromatin Data')

    heatmap2 = axes[1].imshow(sorted_df2, cmap='seismic', aspect='auto', interpolation='none', vmax='10')
    axes[1].set_title('SHAP values')

    heatmap3 = axes[2].imshow(sorted_df3, cmap='seismic', aspect='auto', interpolation='none', vmin='-1', vmax='1')
    axes[2].set_title('Gradient Map')

    # Add colorbars
    fig.colorbar(heatmap1, ax=axes[0], shrink=0.8)
    fig.colorbar(heatmap2, ax=axes[1], shrink=0.8)
    fig.colorbar(heatmap3, ax=axes[2], shrink=0.8)

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
    generate_heatmaps(args.file1, args.file2, args.file3, args.output)

