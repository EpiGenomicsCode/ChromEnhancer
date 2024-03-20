import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_heatmaps(file_paths, output_file):
    # Load the CSV files into pandas DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]

    # Plot the heatmaps
    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 5))

    for i, (df, title) in enumerate(zip(dfs, ["File 1", "File 2", "File 3"])):
        # Set the colormap ranges manually
        if i == 0:
            vmin = 0
            vmax = 0.5
        elif i == 1:
            vmin = -0.0008
            vmax = 0.0008
        elif i == 2:
            vmin = 0
            vmax = 0.23
        im = axes[i].imshow(df, cmap="jet", vmin=vmin, vmax=vmax, interpolation='none')
        axes[i].set_title(title)
        # Add colorbar for each heatmap
        cbar = fig.colorbar(im, ax=axes[i])

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_file)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py file1.csv file2.csv file3.csv output_file.png")
    else:
        file_paths = sys.argv[1:4]
        output_file = sys.argv[4]
        generate_heatmaps(file_paths, output_file)

