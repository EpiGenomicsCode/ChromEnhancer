#!/bin/python
import os, sys, argparse
import pandas as pd
import numpy as np

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description='Rewrite matrix of decimals as matrix of 0/1 values.')

	parser.add_argument('-i','--input', metavar='input_fn', required=True, help='the tab-delimited file of column values (first column is id column)')
	parser.add_argument('-o','--output', metavar='output_fn', required=True, help='the tab-delimited file of rounded column values')
	parser.add_argument('-t','--threshold', metavar='value', type=float, default=0.5, help='the value threshold for calling 0/1')

	args = parser.parse_args()
	return(args)

'''
Coordinate	A549_Rep1_EnhancerScore_SORT	K562_Rep1_EnhancerScore_SORT	K562_Rep2_EnhancerScore_SORT
chr22:0-1000	 0.14789076149463654	 0.023698274046182632	 0.05692596361041069
chr22:1000-2000	 0.1478913128376007	 0.023698274046182632	 0.0569259375333786
chr22:10000-11000	 0.14789657294750214	 0.02369825728237629	 0.056925948709249496
'''

# Main program which takes in input parameters
if __name__ == '__main__':
	'''Collect metadata and EpitopeID results to get detection stats on the YEP data'''

	args = getParams()

	writer = open(args.output, 'w')
	reader = open(args.input, 'r')
	for line in reader:
		# Handle header
		if (line.find('Coordinate')==0):
			writer.write(line)
			continue
		tokens = line.split('\t')
		values = ["1.0" if (float(i)>=args.threshold) else "0.0" for i in tokens[1:]]
		writer.write(tokens[0] + "\t" + '\t'.join(values) + "\n")
	reader.close()
	writer.close()

	# ## Pandas-style implementation

	# # Populate dataframe with tab file data
	# filedata = pd.read_table(args.input, sep='\t')
	# N = len(filedata.columns) - 1

	# # Round scores to 0/1 (is enhancer)
	# # decimals = pd.Series([0] * N, index=filedata.columns[1:])
	# # data = filedata.round(decimals)
	# # filedata = None

	# # Round scores by dynamic threshold
	# data = filedata
	# filedata = None
	# for cid in  data.columns[1:]:
	# 	data.loc[data[cid] >= args.threshold, cid] = 1
	# 	data.loc[data[cid] < args.threshold, cid] = 0

	# # Summarize with sums (total # enhancers per condition)
	# print(data.sum(axis=0))

	# # Write rounded matrix
	# data.to_csv(args.output, sep='\t', index=False)
