import argparse, sys
import pandas as pd
import numpy as np

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Generate a correlation matrix from a tab-delimited matrix file (correlate between columns).
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='the matrix file to correlate')
	parser.add_argument('-o','--output', metavar='tab_fn', required=True, help='the resulting correlation matrix')

	args = parser.parse_args()
	return(args)

def generate_correlation_matrix(input_file, output_file):
	# Read the input file into a pandas DataFrame
	df = pd.read_csv(input_file, delimiter='\t', header=0, index_col=[0])

	# Calculate the correlation matrix
	correlation_matrix = df.corr()

	# Write the correlation matrix to the output file
	correlation_matrix.to_csv(output_file, sep='\t')

# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# To support really large files
	#sys.setrecursionlimit(100000)

	# Usage example
	generate_correlation_matrix(args.input, args.output)
