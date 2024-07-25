import matplotlib.pyplot as plt
import re
import sys
import numpy as np

def plot_box_whisker(matrix_data_file, output_file, y_min=None, y_max=None):
    # Read matrix data from file
    with open(matrix_data_file, 'r') as f:
        matrix_data = f.read()

    # Extract model numbers and auROC values
    models = []
    aurocs = []

    for line in matrix_data.split('\n'):
        match = re.search(r'model(\d+)_.*:(auROC|auPRC): (\d+\.\d+)', line)
        if match:
            model = int(match.group(1))
            auroc = float(match.group(3))
            models.append(model)
            aurocs.append(auroc)

    # Create a dictionary to store data for each model
    model_data = {}
    for model, auroc in zip(models, aurocs):
        if model not in model_data:
            model_data[model] = []
        model_data[model].append(auroc)

    # Convert the dictionary into a list of lists for plotting
    data_to_plot = [model_data[model] for model in sorted(model_data.keys())]

    # Create a box and whisker plot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data_to_plot, labels=sorted(model_data.keys()))

    # Set rainbow colors for boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_to_plot)))
    for i, box in enumerate(bp['boxes']):
        box.set(color=colors[i])

    plt.title('Rainbow-colored Box and Whisker Plot for Different Models')
    plt.xlabel('Model')
    plt.ylabel('auROC')
    plt.grid(True)

    # Set y-axis limits if provided
    if y_min is not None:
        plt.ylim(bottom=float(y_min))
    if y_max is not None:
        plt.ylim(top=float(y_max))

    # Save the plot to an SVG file with 300dpi
    plt.savefig(output_file, dpi=300, format='svg')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python script.py input_matrix_data output_file.svg [y_min] [y_max]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    y_min = sys.argv[3] if len(sys.argv) > 3 else None
    y_max = sys.argv[4] if len(sys.argv) > 4 else None

    plot_box_whisker(input_file, output_file, y_min, y_max)
