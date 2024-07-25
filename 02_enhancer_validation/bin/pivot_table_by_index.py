import argparse
import pandas as pd

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Reformat tab file on metadata from user-selected columns
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='input file to pivot')
	parser.add_argument('-o','--output', metavar='tab_fn', required=True, help='output file after reformatting with pivot')

	parser.add_argument('-x', metavar='col_idx', default=0, type=int, help='row index index for values')
	parser.add_argument('-c', metavar='col_idx', default=1, type=int, help='column index for values')
	parser.add_argument('-v', metavar='col_idx', default=2, type=int, help='values index for values')

	args = parser.parse_args()
	return(args)

def reformat_file(input_file, output_file, index_col=0, column_col=1, values_col=2):
	# Load the data from a tab-delimited file
	data = pd.read_csv(input_file, sep='\t')

	# Get column names
	i_cname = data.columns[index_col]
	c_cname = data.columns[column_col]
	v_cname = data.columns[values_col]

	# Pivot the table to reformat it
	reformatted_data = data.pivot(index=i_cname, columns=c_cname, values=v_cname)

	# Save the reformatted data to a new tab-delimited file
	reformatted_data.to_csv(output_file, sep='\t', na_rep='NaN')

# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# Usage example
	reformat_file(args.input, args.output, args.x, args.c, args.v)
