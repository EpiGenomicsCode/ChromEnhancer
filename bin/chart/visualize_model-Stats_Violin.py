import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import numpy as np
import pandas as pd

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

    # Create a DataFrame from the data
    df = pd.DataFrame({'Model': models, 'auROC': aurocs})

    # Set seaborn style
    sns.set_style('white')
    # Define palette
    palette = 'Set2'

    # Create violin plot
    ax = sns.violinplot(x="Model", y="auROC", data=df, inner=None, dodge=False, palette=palette)

    # Set y-axis limits if provided
    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    if y_max is not None:
        ax.set_ylim(top=float(y_max))

    # Cut violin plot in half
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    # Overlay box plot
    sns.boxplot(x="Model", y="auROC", data=df, ax=ax, width=0.3, saturation=1, showfliers=False,
                boxprops={'zorder': 3, 'facecolor': 'none'})

    old_len_collections = len(ax.collections)
    # Overlay strip plot
    sns.stripplot(x="Model", y="auROC", data=df, jitter=True, dodge=False, palette=palette, ax=ax, edgecolor='black', linewidth=0.5)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))

    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('auROC')
    ax.set_title('Violin Plot with Overlayed Box and Strip Plots')


    # Save the plot
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
