import argparse, sys
import dask.dataframe as dd
import pandas as pd
import fastcluster
import scipy.cluster.hierarchy as sch

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Cluster tab-delimited matrix file.
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='the matrix file to cluster')
	parser.add_argument('-o','--output', metavar='tab_fn', required=True, help='the matrix rewritten in clustered order')

	args = parser.parse_args()
	return(args)



def hierarchical_clustering(file_path, output_matrix_path):
	# Load the data from a tab-delimited file using Dask for large datasets
	data = dd.read_csv(file_path, sep='\t', blocksize="64MB", header=0).compute()

	# Convert to a pandas DataFrame
	data = pd.DataFrame(data)
	data.set_index(data.columns[0], inplace=True)

	# Perform hierarchical clustering on rows using fastcluster for efficiency
	row_linkage = fastcluster.linkage(data, method='average', metric='euclidean')
	row_dendrogram = sch.dendrogram(row_linkage, no_plot=True)
	row_order = row_dendrogram['leaves']

	# Perform hierarchical clustering on columns using fastcluster for efficiency
	col_linkage = fastcluster.linkage(data.T, method='average', metric='euclidean')
	col_dendrogram = sch.dendrogram(col_linkage, no_plot=True)
	col_order = col_dendrogram['leaves']

	# Reorder the data according to the clustering
	clustered_data = data.iloc[row_order, col_order]

	# Save the clustered matrix to a tab-delimited file including headers
	clustered_data.to_csv(output_matrix_path, sep='\t', index=True, header=True)



# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# To support really large files
	sys.setrecursionlimit(100000)

	# Usage example
	hierarchical_clustering(args.input, args.output)
