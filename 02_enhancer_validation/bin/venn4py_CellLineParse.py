import pandas as pd
import matplotlib.pyplot as plt
import argparse
from venny4py.venny4py import *

def parse_input_file(input_file):
    data = {}
    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            for s in tokens[1].split(';'):
                data.setdefault(s,set())
                data[s].add(tokens[0])
    return data

def main():
    parser = argparse.ArgumentParser(description='Generate an 4-way venn diagram plot from a tab-delimited file.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input tab-delimited file: col-1 is item ID and col-2 is semicolon-delimited list of group ids')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for the Venn plot')

    args = parser.parse_args()

    data = parse_input_file(args.input)
    venny4py(sets=data, out=args.output, ext='svg')

if __name__ == '__main__':
    main()

