import argparse, sys
import dask.dataframe as dd
import pandas as pd
import fastcluster
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def getParams():
	'''Parse parameters from the command line'''
	parser = argparse.ArgumentParser(description = """
============
Cluster tab-delimited matrix file and visualize as an image.
============
""", formatter_class = argparse.RawTextHelpFormatter)

	parser.add_argument('-i','--input', metavar='tab_fn', required=True, help='the matrix file to cluster')
	parser.add_argument('-o','--output', metavar='img_fn', required=True, help='the clustered matrix heatmap (.png or .svg)')
	parser.add_argument('-t', '--text_output', metavar='txt_fn', required=True, help='the text file to output the new x-label order')

	args = parser.parse_args()
	return args

def hierarchical_clustering(file_path, output_image_path, text_output_path):
	# Load the data from a tab-delimited file using Dask for large datasets
	data = dd.read_csv(file_path, sep='\t', blocksize="64MB", header=0).compute()

	# Convert to a pandas DataFrame
	data = pd.DataFrame(data)
	data.set_index(data.columns[0], inplace=True)

	# Perform hierarchical clustering on rows using fastcluster for efficiency
	row_linkage = fastcluster.linkage(data, method='average', metric='euclidean')

	# Perform hierarchical clustering on columns using fastcluster for efficiency
	col_linkage = fastcluster.linkage(data.T, method='average', metric='euclidean')

	# Create a clustermap using seaborn
	g = sns.clustermap(data, row_linkage=row_linkage, col_linkage=col_linkage, cmap='viridis', xticklabels=True, yticklabels=True)

	# Rotate the x-axis labels for better readability
	plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x labels if needed
	
	plt.savefig(output_image_path, bbox_inches='tight')

	# Output the new x-label order to a text file
	new_x_labels = [label.get_text() for label in g.ax_heatmap.get_xticklabels()]
	with open(text_output_path, 'w') as f:
		for label in new_x_labels:
			f.write(f"{label}\n")


# Main program which takes in input parameters
if __name__ == '__main__':
	args = getParams()

	# To support really large files
	sys.setrecursionlimit(100000)

	# Usage example
	hierarchical_clustering(args.input, args.output, args.text_output)
