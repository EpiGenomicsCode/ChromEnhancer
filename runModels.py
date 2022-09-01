from Chrom_Proj.runner import runner
from Chrom_Proj.chrom_dataset import Chromatin_Dataset
import torch
import Chrom_Proj.visualizer as v


# TODO train model and save validation output, send validation output to William in corresponding bed file col before the . for 
# correct validation & and PRC do multiple models on multiple data

def main():
    chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]
    epochs = 10
    batch_size = 32

    runner(chromtypes,  
            id="A549", 
            trainLabel="chr10-chr17", 
            testLabel="chr10", 
            validLabel="chr17",
            epochs=epochs, batchSize=batch_size,
            fileLocation="./Data/220802_DATA", 
            modelType=1
            )

    runner(chromtypes,  
        id="A549", 
        trainLabel="chr10-chr17", 
        testLabel="chr10", 
        validLabel="chr17",
        epochs=epochs, batchSize=batch_size,
        fileLocation="./Data/220802_DATA", 
        modelType=2
        )
        
    # runner(chromtypes,  
    #         id="A549", 
    #         trainLabel="chr10-chr17", 
    #         testLabel="chr10", 
    #         validLabel="chr17",
    #         epochs=epochs, batchSize=batch_size,
    #         fileLocation="./Data/220802_DATA", 
    #         modelType=3
    #         )


    # runner(chromtypes,  
    #         id="HepG2", 
    #         trainLabel="chr10-chr17", 
    #         testLabel="chr10", 
    #         validLabel="chr17",
    #         epochs=epochs, batchSize=batch_size,
    #         fileLocation="./Data/220802_DATA", 
    #         modelType=1
    #         )

    # runner(chromtypes,  
    #         id="HepG2", 
    #         trainLabel="chr10-chr17", 
    #         testLabel="chr10", 
    #         validLabel="chr17",
    #         epochs=epochs, batchSize=batch_size,
    #         fileLocation="./Data/220802_DATA", 
    #         modelType=2
    #         )

    # runner(chromtypes,  
    #         id="HepG2", 
    #         trainLabel="chr10-chr17", 
    #         testLabel="chr10", 
    #         validLabel="chr17",
    #         epochs=epochs, batchSize=batch_size,
    #         fileLocation="./Data/220802_DATA", 
    #         modelType=3
    #         )

    # runner(chromtypes,  
    #     id="K562", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=1
    #     )

    # runner(chromtypes,  
    #     id="K562", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=2
    #     )
    
    # runner(chromtypes,  
    #     id="K562", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=3
    #     )

    # runner(chromtypes,  
    #     id="MCF7", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=1
    #     )    
    
    # runner(chromtypes,  
    #     id="MCF7", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=2
    #     )  

    # runner(chromtypes,  
    #     id="MCF7", 
    #     trainLabel="chr10-chr17", 
    #     testLabel="chr10", 
    #     validLabel="chr17",
    #     epochs=epochs, batchSize=batch_size,
    #     fileLocation="./Data/220802_DATA", 
    #     modelType=3
    #     )    


main()
