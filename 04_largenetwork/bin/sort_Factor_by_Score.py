import argparse

def load_file(file_path):
    """Loads a file and returns a list of its lines."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def main():
    parser = argparse.ArgumentParser(description="Sort IDs by scores.")
    parser.add_argument('id_file', type=str, help="Path to the file containing the list of IDs.")
    parser.add_argument('score_file', type=str, help="Path to the file containing the list of scores.")
    parser.add_argument('output_file', type=str, help="Path to the output file where sorted IDs and scores will be saved.")
    
    args = parser.parse_args()

    ids = load_file(args.id_file)
    scores = list(map(float, load_file(args.score_file)))

    if len(ids) != len(scores):
        raise ValueError("The number of IDs and scores must match.")

    # Combine IDs and scores, sort by scores in descending order
    sorted_pairs = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)

    # Write the sorted IDs and scores to the output file
    with open(args.output_file, 'w') as file:
        for id_, score in sorted_pairs:
            file.write(f"{id_}\t{score}\n")

if __name__ == "__main__":
    main()

