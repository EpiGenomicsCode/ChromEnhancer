import sys

def parse_input(input_file):
    data = {}
    with open(input_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            cell_line, mark, quality = key.split('_')
            if mark == "GROseq" or mark == "PROseq":
                mark = "RNA"
            if cell_line not in data:
                data[cell_line] = {}
            if mark not in data[cell_line]:
                data[cell_line][mark] = {}
            data[cell_line][mark][quality] = value
    return data

def generate_output(data):
    cell_lines = sorted(data.keys())
    marks = sorted(set(mark for cell_line in data.values() for mark in cell_line.keys()))
    qualities = sorted(list(data[list(data.keys())[0]][marks[0]].keys()))

    output = '\t' + '\t'.join(cell_lines) + '\n'
    CELL = ["A549", "HepG2", "K562", "MCF7"]
    TARGET = ["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3", "CTCF", "p300", "POLR2A", "RNA"]
    LABEL = ["Stringent", "Lenient", "Random"]

    output = "\t" + CELL[0] + "\t" + CELL[0] + "\t" + CELL[0] + "\t" + CELL[1] + "\t" + CELL[1] + "\t" + CELL[1] +"\t" + CELL[2] + "\t" + CELL[2] + "\t" + CELL[2] +"\t" + CELL[3] + "\t" + CELL[3] + "\t" + CELL[3] + "\n"
    for mark in TARGET:
        output += mark
        for cell in CELL:
            for label in LABEL:
                value = data[cell].get(mark, {}).get(label, '')
                output += "\t" + value
        output += "\n"
#                print("\t" + label)

    return output

#    for mark in marks:
#        print(mark)
#        for quality in qualities:
#            output += mark + '\t' + quality + '\t'
#            for cell_line in cell_lines:
#                value = data[cell_line].get(mark, {}).get(quality, '')
#                output += value + '\t'
#            output += '\n'
#    return output

def main(input_file, output_file):
    data = parse_input(input_file)
    output = generate_output(data)
    
    with open(output_file, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)

