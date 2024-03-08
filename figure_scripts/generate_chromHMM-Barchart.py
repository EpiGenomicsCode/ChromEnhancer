import matplotlib.pyplot as plt
import sys

def read_data(input_file):
    data = {}
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            label = parts[0].strip()
            value = float(parts[1].split()[1])
            data[label] = value
    return data

def create_bar_chart(data, output_file):
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots()
    ax.barh(labels[::-1], values[::-1])  # Reverse the order

    ax.set_xlabel('% Overlap')
    ax.set_title('chromHMM Label')

    fig.tight_layout()
    fig.savefig(output_file, format='svg', dpi=300)  # Save as SVG at 300 DPI

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_chromHMM-Barchart.py chromHMM-Freq.tab chromHMM-barchart.svg")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = read_data(input_file)
    create_bar_chart(data, output_file)

