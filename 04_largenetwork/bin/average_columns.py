import pandas as pd
import argparse

def calculate_column_averages(input_file, output_file):
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t', header=None)
    
    # Calculate the average across all rows for each column
    averages = df.mean(axis=0)
    
    # Convert the result to a DataFrame with a single row
    result = pd.DataFrame([averages]).T
    
    # Write the result to a new TSV file
    result.to_csv(output_file, sep='\t', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Calculate the average across all rows for each column in a TSV file and output to a new TSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file.')
    parser.add_argument('output_file', type=str, help='Path to the output TSV file.')
    
    args = parser.parse_args()
    
    calculate_column_averages(args.input_file, args.output_file)

if __name__ == '__main__':
    main()

