import pandas as pd
import matplotlib.pyplot as plt
import argparse

def generate_heatmap(input_file_path, heatmap_file_path):
    # Load the processed data into a DataFrame
    data = pd.read_csv(input_file_path, index_col=0)

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', interpolation='none')
    plt.title("Heatmap of IsEnriched Values")
    plt.xlabel("Cell Line")
    plt.ylabel("ChIP-CellLine - TF-Target")
    #plt.tight_layout()

    # Save the heatmap
    plt.savefig(heatmap_file_path, format='svg')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmap from CSV file")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('heatmap_file', type=str, help='Path to save the heatmap image file')
    args = parser.parse_args()
    
    generate_heatmap(args.input_file, args.heatmap_file)
