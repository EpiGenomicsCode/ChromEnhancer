import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Argument parser setup
parser = argparse.ArgumentParser(description='Plot heatmap from TSV file.')
parser.add_argument('input_tsv', type=str, help='Path to the input TSV file')
parser.add_argument('output_path', type=str, help='Directory to save the output plots')

args = parser.parse_args()

# Read TSV file into a DataFrame
df = pd.read_csv(args.input_tsv, sep='\t', header=None)  # Assuming no header in the TSV file
#print(df)

# Calculate mean and standard deviation
mean = df.values.mean()
std_dev = df.values.std()
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

compressed_activation = df.to_numpy()
# Plot the heatmap
plt.figure(figsize=(6, 6))
plt.imshow(compressed_activation.reshape(len(compressed_activation), -1), cmap="seismic", vmin=mean - (std_dev / 1), vmax=mean + (std_dev / 1), aspect='auto', interpolation='none')
plt.colorbar()
plt.title(f"Compressed Integrated Gradient")

# Save the plots
plt.savefig(f"{args.output_path}/heatmap_grad_compressed.svg", format='svg')
plt.savefig(f"{args.output_path}/heatmap_grad_compressed.png", format='png')
plt.close()

