#!/bin/python
import os, sys, argparse
import pandas as pd
import numpy as np

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description='Merge and replace two columns in a tab-delimited file.')

	parser.add_argument('-i','--input', metavar='input_fn', required=True, help='the tab-delimited file')
	parser.add_argument('-o','--output', metavar='output_fn', required=True, help='the tab-delimited output file')
	parser.add_argument('-a','--col-a', metavar='value', required=True, type=int, help='the first 0-indexed column to merge')
	parser.add_argument('-b','--col-b', metavar='value', required=True, type=int, help='the second 0-indexed column to merge')

	# select a metric for determining the value of the merged column
	metric_group = parser.add_mutually_exclusive_group()
	metric_group.add_argument('--intersect', action='store_true')
	metric_group.add_argument('--union', action='store_true')

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
		tokens = line.strip().split('\t')
		a_val = tokens[args.col_a]
		b_val = tokens.pop(args.col_b)
		# Handle header
		if (line.find('Coordinate')==0):
			tokens[args.col_a] = a_val + "_X_" + b_val
		else:
			if (args.intersect):
				tokens[args.col_a] = "1.0" if (float(b_val)==1 and float(a_val)==1) else "0.0"
			if (args.union):
				tokens[args.col_a] = "1.0" if (float(b_val)==1 or float(a_val)==1) else "0.0"
		writer.write('\t'.join(tokens) + "\n")
	reader.close()
	writer.close()