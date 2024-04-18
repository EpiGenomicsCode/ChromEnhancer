import sys
import re
from collections import defaultdict

def calculate_averages(input_file, output_file):
    averages = defaultdict(list)

    with open(input_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            key_parts = re.split('_|-', key)
            unique_key = "_".join([key_parts[0], key_parts[1], key_parts[3]])
            averages[unique_key].append(float(value))

    with open(output_file, 'w') as f:
        for key, values in averages.items():
            avg = sum(values) / len(values)
            f.write(f"{key}\t{avg}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    calculate_averages(input_file, output_file)

