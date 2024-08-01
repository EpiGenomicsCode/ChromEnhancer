import pandas as pd
import argparse

def main(input_file_path, output_file_path):
    # Load the data into a DataFrame
    data = pd.read_csv(input_file_path, sep="\t")
    print("Columns in the input file:", data.columns)

    # Create the output DataFrame with columns for each cell line
    cell_lines = ['A549', 'HepG2', 'K562', 'MCF7']
    output = pd.DataFrame(0, index=data['ChIP-CellLine'] + '-' + data['TF-Target'], columns=cell_lines)

    print("Columns in the input file:", output.columns)

    # Create a dictionary to keep track of the combinations
    index_dict = {}
    for idx, row in data.iterrows():
        index = row['ChIP-CellLine'] + '-' + row['TF-Target']
        if index not in index_dict:
            index_dict[index] = {cell: 0 for cell in ['A549', 'HepG2', 'K562', 'MCF7']}
        if row['IsEnriched'] == 1:
            index_dict[index][row['Enh-CellLine']] = 1

    # Convert the dictionary to a DataFrame
    output = pd.DataFrame.from_dict(index_dict, orient='index')

    # Calculate the sum of IsEnriched values
    output['SumIsEnriched'] = output.sum(axis=1)
    # Filter out rows where the sum is 0
    output = output[output['SumIsEnriched'] > 0]
    # Add the 'ChIP-CellLine' + '-' + 'TF-Target' index as a column for sorting
    output['Index'] = output.index
    
    # Sort by the sum of the IsEnriched values and then by cell line ID
    output = output.sort_values(by=['SumIsEnriched', 'MCF7', 'K562', 'HepG2', 'A549', 'Index'], ascending=[False, True, True, True, True, True])
    output = output.drop(columns=['SumIsEnriched', 'Index'])

    # Save the final DataFrame to a new CSV file
    output.to_csv(output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV file format")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_file', type=str, help='Path to save the output CSV file')
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)

