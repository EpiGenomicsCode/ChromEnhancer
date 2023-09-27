#!/bin/python
import os, sys, argparse
import pandas as pd
import numpy as np

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description='Serial outer merge of a list of tab-delimited matrix files and fill in missing values with 0.')

	parser.add_argument('-i','--input', metavar='input_fn', required=True, nargs='+', help='the list of input files formatted with id in the first column and col headers in the first row')
	parser.add_argument('-o','--output', metavar='output_fn', required=True, help='the tab-delimited file of the merged matrix (each column named by original input filenames)')

	args = parser.parse_args()
	return(args)

# Main program which takes in input parameters
if __name__ == '__main__':
	'''Collect metadata and EpitopeID results to get detection stats on the YEP data'''

	args = getParams()

	# Load first file
	data = pd.read_table(args.input[0], sep='\t', index_col=0, header=0)
	# data.columns = [args.input[0]]

	# Add rest of files
	for filename in args.input[1:]:
		# Load next into df
		other = pd.read_table(filename, sep='\t', index_col=0, header=0)
		# other.columns = [filename]
		# Join dfs
		data = data.join(other, how='outer')

	# Fill NaNs
	data = data.fillna(0)

	# Write rounded matrix
	data.to_csv(args.output, sep='\t')
