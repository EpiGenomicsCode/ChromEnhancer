import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Function to count the number of lines in a file
def count_lines(filepath):
    with open(filepath, 'r') as file:
        return sum(1 for _ in file)

# Main function to process files and generate the plot
def main(input_folder, output_file):
    # Dictionaries to store the line counts
    enh_counts = {}
    rand_counts = {}

    # Iterate over each file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tsv'):
            # Count the number of lines in the file
            file_path = os.path.join(input_folder, filename)
            line_count = count_lines(file_path)
            
            # Determine if the file is Enh or Rand
            if 'Enh' in filename:
                enh_counts[filename] = line_count
            elif 'Rand' in filename:
                rand_counts[filename] = line_count

    # Calculate the log2 ratios
    labels = []
    log2_ratios = []
    for enh_file in enh_counts:
        base_name = enh_file.replace('-Enh.tsv', '')
        rand_file = base_name + '-Rand.tsv'
        if rand_file in rand_counts:
            enh_count = enh_counts[enh_file]
            rand_count = rand_counts[rand_file]
            log2_ratio = np.log2(enh_count / rand_count)
            labels.append(base_name)
            log2_ratios.append(log2_ratio)

    # Sort the labels and corresponding log2_ratios alphabetically by label
    sorted_indices = np.argsort(labels)
    labels = [labels[i] for i in sorted_indices]
    log2_ratios = [log2_ratios[i] for i in sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(labels, log2_ratios, color='skyblue')
    plt.ylabel('Log2(Enhancer / Random)')
    plt.title('Log2 Ratio of Enhancer to Random for ChIA-PET')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot to the specified output file
    plt.savefig(output_file, format='svg')

# Argparse to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bar chart of log2 ratios for ChIA data")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing .tsv files")
    parser.add_argument('output_file', type=str, help="Path to save the output bar chart image")

    args = parser.parse_args()
    
    # Run the main function with provided arguments
    main(args.input_folder, args.output_file)

