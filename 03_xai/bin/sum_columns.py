import pandas as pd
import argparse

def sum_columns_in_chunks(input_file, output_file, chunk_size=100):
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t', header=None)
    
    # Determine the number of chunks
    num_chunks = df.shape[1] // chunk_size
    print("Chunks detected: " + str(num_chunks))
    # Sum every 'chunk_size' columns
    result = pd.DataFrame()
    for i in range(num_chunks):
        chunk_sum = df.iloc[:, i*chunk_size:(i+1)*chunk_size].sum(axis=1)
        result[f'Sum_{i+1}'] = chunk_sum
    
    # Write the result to a new TSV file
    result.to_csv(output_file, sep='\t', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Sum every 100 columns of a TSV file and output to a new TSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file.')
    parser.add_argument('output_file', type=str, help='Path to the output TSV file.')
    
    args = parser.parse_args()
    
    sum_columns_in_chunks(args.input_file, args.output_file)

if __name__ == '__main__':
    main()

