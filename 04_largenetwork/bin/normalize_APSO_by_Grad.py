import argparse
import pandas as pd

def read_tsv(file_path):
    """Reads a TSV file and returns a DataFrame."""
    return pd.read_csv(file_path, sep='\t', header=None)

def multiply_matrices(matrix1, matrix2):
    """Multiplies two matrices element-wise."""
    return matrix1 * matrix2

def write_tsv(matrix, output_file_path):
    """Writes a DataFrame to a TSV file."""
    matrix.to_csv(output_file_path, sep='\t', header=False, index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Multiply corresponding values from two TSV files.')
    parser.add_argument('file1', type=str, help='Path to the first TSV file.')
    parser.add_argument('file2', type=str, help='Path to the second TSV file.')
    parser.add_argument('output_file', type=str, help='Path to the output TSV file.')

    # Parse arguments
    args = parser.parse_args()

    # Read TSV files
    matrix1 = read_tsv(args.file1)
    matrix2 = read_tsv(args.file2)

    # Multiply matrices
    result_matrix = multiply_matrices(matrix1, matrix2)

    # Write result to output file
    write_tsv(result_matrix, args.output_file)

if __name__ == '__main__':
    main()
