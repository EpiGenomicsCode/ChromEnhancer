import os
import argparse
import pandas as pd

def split_file_by_column(file_path, output_dir, column_index):
    # Read the file using pandas
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group by the specified column and write each group to a separate file
    for category, group in df.groupby(df.columns[column_index]):
        output_file_path = os.path.join(output_dir, f"{category}.tsv")
        group.to_csv(output_file_path, sep='\t', header=False, index=False)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Split a file into multiple files based on a categorical value in a specified column.')
    parser.add_argument('file_path', type=str, help='Path to the input file')
    parser.add_argument('output_dir', type=str, help='Directory to store the output files')
    parser.add_argument('column_index', type=int, help='Zero-based index of the column to split by')
    
    args = parser.parse_args()
    
    # Call the split function with the parsed arguments
    split_file_by_column(args.file_path, args.output_dir, args.column_index)

if __name__ == "__main__":
    main()

