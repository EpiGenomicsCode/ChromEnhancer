import argparse
import csv

def filter_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        header = ["chr1" , "start1", "start2", "chr2", "start2", "end2"]
        csv_writer.writerow(header)
        for line in infile:
            # Split the line into columns
            columns = line.strip().split()
            
            # Check the conditions
            if (int(columns[3]) > 0 and int(columns[15]) > 0) or (int(columns[7]) > 0 and int(columns[11]) > 0):
                if int(columns[3]) > 0:
                    columns[0] = "GENE" + columns[0]
                    columns[8] = "ENH" + columns[8]
                else:
                    columns[0] = "ENH" + columns[0]
                    columns[8] = "GENE" + columns[8]
                selected_columns = [columns[0], columns[1], columns[2], columns[8], columns[9], columns[10]]
                csv_writer.writerow(selected_columns)

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Filter lines from input file and write to output CSV file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output CSV file")

    args = parser.parse_args()

    # Call the filter function with provided arguments
    filter_lines(args.input_file, args.output_file)

    print(f"Filtered lines have been written to {args.output_file}.")

