import torch 
import pandas as pd
from torch import nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split as rnd_splt


from Model.chrom_dataset import Chromatin_Dataset
from Model.model import Chromatin_Network
from Model.util import fitSVM, plotPCA, trainModel


def main():
   
   # Load the training  data
   c10_17_ctf_1     = Chromatin_Dataset(chromType="chr10-chr17", chromName="CTCF-1")
   c10_17_ctf_2     = Chromatin_Dataset(chromType="chr10-chr17", chromName="CTCF-2")
   c10_17_H3K27ac_1 = Chromatin_Dataset(chromType="chr10-chr17", chromName="H3K27ac-1")
   c10_17_H3K27ac_2    = Chromatin_Dataset(chromType="chr10-chr17", chromName="H3K27ac-2")
   c10_17_H3K4me3_1    = Chromatin_Dataset(chromType="chr10-chr17", chromName="H3K4me3-1")
   c10_17_H3K4me3_2    = Chromatin_Dataset(chromType="chr10-chr17", chromName="H3K4me3-2")
   c10_17_p300_1    = Chromatin_Dataset(chromType="chr10-chr17", chromName="p300-1")
   c10_17_p300_2    = Chromatin_Dataset(chromType="chr10-chr17", chromName="p300-2")
   c10_17_PolII_1    = Chromatin_Dataset(chromType="chr10-chr17", chromName="PolII-1")
   c10_17_PolII_2    = Chromatin_Dataset(chromType="chr10-chr17", chromName="PolII-2")

   # Load the testing data and validation
   HOLD_c10_17_ctf_1     = Chromatin_Dataset(chromType="chr10", chromName="CTCF-1", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_ctf_2     = Chromatin_Dataset(chromType="chr17", chromName="CTCF-2", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_H3K27ac_1     = Chromatin_Dataset(chromType="chr10", chromName="H3K27ac-1", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_H3K27ac_2     = Chromatin_Dataset(chromType="chr17", chromName="H3K27ac-2", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_H3K4me3_1     = Chromatin_Dataset(chromType="chr10", chromName="H3K4me3-1", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_H3K4me3_2     = Chromatin_Dataset(chromType="chr17", chromName="H3K4me3-2", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_p300_1     = Chromatin_Dataset(chromType="chr10", chromName="p300-1", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_p300_2     = Chromatin_Dataset(chromType="chr17", chromName="p300-2", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_PolII_1     = Chromatin_Dataset(chromType="chr10", chromName="PolII-1", file_location="./Data/220627_DATA/HOLDOUT/*")
   HOLD_c10_17_PolII_2     = Chromatin_Dataset(chromType="chr17", chromName="PolII-2", file_location="./Data/220627_DATA/HOLDOUT/*")
   
   trainer   = [c10_17_ctf_1      , c10_17_ctf_2,c10_17_H3K27ac_1,c10_17_H3K27ac_2,c10_17_H3K4me3_1, c10_17_H3K4me3_2, c10_17_p300_1,c10_17_p300_2, c10_17_PolII_1,c10_17_PolII_2 ]
   tester    = [HOLD_c10_17_ctf_1 , HOLD_c10_17_H3K27ac_1, HOLD_c10_17_H3K4me3_1]
   validator = [HOLD_c10_17_p300_1, HOLD_c10_17_PolII_1]#[HOLD_c10_17_ctf_2, HOLD_c10_17_H3K27ac_2, HOLD_c10_17_H3K4me3_2, HOLD_c10_17_p300_2, HOLD_c10_17_PolII_2]
   
   # Train the SVM
   # supportvectormachine = svm.SVC(verbose=True, tol=1e-1,cache_size=1024, max_iter=epochs, kernel="poly", degree=7)
   # supportvectormachine = fitSVM(supportvectormachine, epochs, trainer, tester, validator)

   # # PCA plot
   # for t in trainer:
   #    plotPCA(t)
   # for t in tester:
   #    plotPCA(t)
   # for t in validator:
   #    plotPCA(t)


   # paramets
   
   # Detect GPU or CPU
   epochs = 10
   batch_size = 32
   learning_rate = 1e-3
   inputSize = 100

   # Build the model 
   model = Chromatin_Network(input_shape=inputSize)
   print(model)

   # Compile the model
   optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
   loss_fn = nn.BCEWithLogitsLoss()

   trainModel(trainer, tester, validator, model, optimizer, loss_fn, batch_size, epochs)
   

main()