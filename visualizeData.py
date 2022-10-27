import Chrom_Proj.visualizer as v

# TODO Implement
def main():
    models = [1,2,3]
    for i in models:
        v.plotModel(i)
    v.modelPreformance("1","A549")

main()