import sys
import gzip

def replace_scores(bed_file, score_file, output_file):
    # Read scores from the single-column text file, skipping the header
    with open(score_file, 'r') as sf:
        scores = sf.readlines()[1:]  # Skip the header
    
    # Ensure there are no extra newlines
    scores = [score.strip() for score in scores]
    
    # Read the GZipped BED file and replace scores
    with gzip.open(bed_file, 'rt') as bf:
        bed_lines = bf.readlines()
    
    if len(scores) != len(bed_lines):
        print(len(scores))
        print(len(bed_lines))
        raise ValueError("The number of scores does not match the number of lines in the BED file.")
    
    # Create the output with replaced scores
    with open(output_file, 'w') as of:
        for i, line in enumerate(bed_lines):
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                parts[4] = scores[i]
            of.write('\t'.join(parts) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <gzipped_bed_file> <score_file> <output_file>")
        sys.exit(1)

    bed_file = sys.argv[1]
    score_file = sys.argv[2]
    output_file = sys.argv[3]

    replace_scores(bed_file, score_file, output_file)
    print(f"Scores replaced successfully. Output written to {output_file}.")
