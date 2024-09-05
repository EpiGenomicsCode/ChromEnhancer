import pandas as pd
from scipy.stats import pearsonr
import argparse

# Function to calculate Pearson correlation between two rows
def calculate_pearson(row1, row2):
    return pearsonr(row1, row2)[0]

def main(multi_row_file, single_row_file, output_file):
    # Read the TSV files
    multi_df = pd.read_csv(multi_row_file, sep='\t', header=None)
    single_df = pd.read_csv(single_row_file, sep='\t', header=None)

    # Ensure the single-row file has exactly one row
    assert single_df.shape[0] == 1, "The single-row file must contain exactly one row."

    single_row = single_df.iloc[0].values

    # Calculate Pearson correlation for each row in the multi-row file
    correlation_scores = multi_df.apply(lambda row: calculate_pearson(row.values, single_row), axis=1)

    # Save the scores to the output file
    correlation_scores.to_csv(output_file, sep='\t', header=False, index=False)

    print(f"Pearson correlation scores saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Pearson correlation between rows of two TSV files.')
    parser.add_argument('multi_row_file', type=str, help='Path to the multi-row TSV file')
    parser.add_argument('single_row_file', type=str, help='Path to the single-row TSV file')
    parser.add_argument('output_file', type=str, help='Path to the output file to save the correlation scores')

    args = parser.parse_args()

    main(args.multi_row_file, args.single_row_file, args.output_file)

