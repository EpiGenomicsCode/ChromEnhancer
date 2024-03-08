import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Function to read CSV files from a folder and calculate average
def read_and_average_csv(folder_path):
    # Initialize an empty DataFrame to store the aggregated data
    aggregated_data = pd.DataFrame()

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path, sep=',', header=None, index_col=0, names=['index', filename])
            # Add data to aggregated DataFrame
            aggregated_data = pd.concat([aggregated_data, df.iloc[:, 0]], axis=1)

    # Calculate the mean along columns (axis=1) to get the average
    average_data = aggregated_data.mean(axis=1)
    return average_data, aggregated_data

# Function to plot average line and individual component vectors
def plot_data(average_data, aggregated_data, output_file):
    # Plot the average line
    plt.plot(average_data, color='blue', label='Average', linewidth=2)

    # Plot individual component vectors with alpha=0.2
    for col in aggregated_data.columns:
        plt.plot(aggregated_data[col], color='grey', alpha=0.2)

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Average and Individual Component Vectors')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate a line plot of the average of CSV files with individual component vectors shown in the background.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing CSV files')
    parser.add_argument('output_png', type=str, help='Path to the output PNG file')
    args = parser.parse_args()

    # Read and average CSV files
    average_data, aggregated_data = read_and_average_csv(args.input_folder)

    # Plot the data and save to output PNG file
    plot_data(average_data, aggregated_data, args.output_png)

