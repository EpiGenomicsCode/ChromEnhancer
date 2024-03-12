import matplotlib.pyplot as plt
import sys

def read_data(input_files):
    all_data = []
    for input_file in input_files:
        data = {}
        with open(input_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')  # Use tab delimiter
                label = parts[0].strip()
                value = float(parts[1])
                data[label] = value
        all_data.append(data)
    return all_data

def create_bar_chart(all_data, input_files, output_file, legend_file):
    labels = list(all_data[0].keys())[::-1]  # Reverse order of y-axis labels
    num_series = len(all_data)

    fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size

    width = 0.2
    for i, data in enumerate(reversed(all_data)):  # Iterate over reversed data series
        values = [data[label] for label in labels]  # Match order with reversed labels
        positions = [j + i * width for j in range(len(labels))]
        ax.barh(positions, values, height=width, label=input_files[num_series - i - 1])  # Use input file name as label

    ax.set_xlabel('% Overlap')
    ax.set_title('chromHMM Label')
    ax.set_yticks([pos + width*num_series/2 for pos in range(len(labels))])
    ax.set_yticklabels(labels)

    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout parameters

    fig.savefig(output_file, format='svg', dpi=300)  # Save as SVG at 300 DPI

    # Create a separate figure for the legend
    legend_fig = plt.figure(figsize=(2, 6))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend_fig.legend(*ax.get_legend_handles_labels(), loc='center')
    legend_fig.savefig(legend_file, format='svg', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python generate_chromHMM-Barchart.py input_file1 input_file2 input_file3 input_file4 output_file.svg legend_file.svg")
        sys.exit(1)

    input_files = sys.argv[1:5]
    output_file = sys.argv[5]
    legend_file = sys.argv[6]

    all_data = read_data(input_files)
    create_bar_chart(all_data, input_files, output_file, legend_file)

