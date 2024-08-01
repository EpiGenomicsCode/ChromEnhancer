import argparse
import pandas as pd
import scipy.stats as stats
#from scipy.stats import poisson_means_test
from scipy.stats import false_discovery_control

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Perform Poisson means test for each row of tf-bound table
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='input file with rows of contingency table info')
	parser.add_argument('-o','--output', metavar='tab_fn', required=True, help='output file with chi square statistics')

	args = parser.parse_args()
	return(args)

def poisson_means_test(input_file, output_file):
	# Load the data from a tab-delimited file
	data = pd.read_csv(input_file, sep='\t', header=0)

	# Initialize a list to store the results
	results = []

	# Perform Chi-square test on each row
	for index, row in data.iterrows():
		# Calculate the stats
		#k2 = int((row.iloc[6] / row.iloc[4]) * row.iloc[9])
		k2 = int(row.iloc[9])
		n1 = int(row.iloc[6])
		k1 = int(row.iloc[8])
		n2 = int(row.iloc[5])
		#print(k1)
		#print(n1)
		#print(k2)
		#print(n2)
		# Perform the Poisson means test
		pStat, pVal = stats.poisson_means_test(k1, n1, k2, n2, alternative='greater')

		# Append the results
		results.append({"Index": index, data.columns[0]: row.iloc[0], data.columns[1]: row.iloc[1], data.columns[2]: row.iloc[2], data.columns[3]: row.iloc[3], "Poisson": pStat, "P-value": pVal})

	# Convert results to a DataFrame
	results_df = pd.DataFrame(results)

	# Perform Benjamini-Hochberg correction
	results_df['CorrectedP-value'] = false_discovery_control(results_df["P-value"], method='by')

	# Call Enrichment based on corrected p-value
	#results_df['Enriched'] = np.where(results_df['CorrectedP-value']<0.05, 'Yes', 'No')
	results_df['IsEnriched'] = [1 if p < 0.05 else 0 for p in results_df['CorrectedP-value'] ]

	# Save the results to a tab-delimited file
	results_df.to_csv(output_file, sep='\t', index=False)


# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# Usage example
	poisson_means_test(args.input, args.output)
