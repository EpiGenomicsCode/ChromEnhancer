import Chrom_Proj.visualizer as v

def main():
    ids = ["A549", "HepG2", "K562", "MCF7" ]
    for id in ids:
        v.plotAll("output/coord/", id)

main()