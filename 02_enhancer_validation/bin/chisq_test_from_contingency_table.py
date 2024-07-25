import argparse
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import false_discovery_control

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Perform Chi Square test for each row of contingency info
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='input file with rows of contingency table info')
	parser.add_argument('-o','--output', metavar='tab_fn', required=True, help='output file with chi square statistics')

	parser.add_argument('-nn', metavar='n_column', default=2, type=int, help='Not A and Not B')
	parser.add_argument('-an', metavar='n_column', default=3, type=int, help='A but not B')
	parser.add_argument('-nb', metavar='n_column', default=4, type=int, help='B but not A')
	parser.add_argument('-ab', metavar='n_column', default=5, type=int, help='Both A and B')

	args = parser.parse_args()
	return(args)

def chi_square_test(input_file, output_file):
	# Load the data from a tab-delimited file
	data = pd.read_csv(input_file, sep='\t', header=0)

	# Initialize a list to store the results
	results = []

	# Perform Chi-square test on each row
	for index, row in data.iterrows():
		# Create the contingency table
		contingency_table = [
			[row.iloc[args.nn], row.iloc[args.an]],
			[row.iloc[args.nb], row.iloc[args.ab]]
		]

		# Perform the Chi-square test
		chi2, p, _, _ = chi2_contingency(contingency_table)

		# Append the results
		results.append({"Index": index, data.columns[0]: row.iloc[0], data.columns[1]: row.iloc[1], "Chi2": chi2, "P-value": p})

	# Convert results to a DataFrame
	results_df = pd.DataFrame(results)

	# Perform Benjamini-Hochberg correction
	results_df['CorrectedP-value'] = false_discovery_control(results_df["P-value"], method='bh')

	# Call Enrichment based on corrected p-value
	#results_df['Enriched'] = np.where(results_df['CorrectedP-value']<0.05, 'Yes', 'No')
	results_df['IsEnriched'] = [1 if p < 0.05 else 0 for p in results_df['CorrectedP-value'] ]

	# Save the results to a tab-delimited file
	results_df.to_csv(output_file, sep='\t', index=False)


# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# Usage example
	chi_square_test(args.input, args.output)
