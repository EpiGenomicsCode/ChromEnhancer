import sys
import numpy as np
import matplotlib.pyplot as plt

def read_matrix(input_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            rows = []
            columns = []
            data = []
            for line in lines:
                parts = line.strip().split()
                if not columns:
                    columns = parts[1:]
                else:
                    rows.append(parts[0])
                    data.append([float(x) for x in parts[1:]])
            return rows, columns, np.array(data)
    except FileNotFoundError:
        print("Input file not found.")
        sys.exit(1)
    except Exception as e:
        print("An error occurred while reading the input file:", e)
        sys.exit(1)

def plot_heatmap(rows, columns, data, output_file, vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='plasma', interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title('Heatmap')
    plt.xticks(np.arange(len(columns)), columns, rotation=45)
    plt.yticks(np.arange(len(rows)), rows)
    plt.tight_layout()
    plt.savefig(output_file, format='svg')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Define manual minimum and maximum values
    vmin = 0  # set your minimum value here
    vmax = 10000  # set your maximum value here

    rows, columns, data = read_matrix(input_file)
    plot_heatmap(rows, columns, data, output_file, vmin=vmin, vmax=vmax)
    print("Heatmap saved as", output_file)

