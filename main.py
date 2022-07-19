import torch
import pandas as pd
from torch import nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split as rnd_splt

from Model.runner import getK562
from Model.model import Chromatin_Network
from Model.util import fitSVM, plotPCA, runModel, validate, loadModel

# TODO train model and save validation output, send validation output to William in corresponding bed file col before the . for 
# correct validation & and PRC do multiple models on multiple data

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chromtypes = ["CTCF-1", "H3K4me3-1", "H3K27ac-1", "p300-1", "PolII-1"]

    trainer, tester, validator = getK562(chromtypes)
    modelK562 = Chromatin_Network("K562")
    runModel(
                trainer,
                tester,
                validator,
                model=modelK562,
                optimizer=torch.optim.Adam(modelK562.parameters(), lr=1e-4),
                loss_fn=nn.BCEWithLogitsLoss(),
                batch_size=64,
                epochs=10
            )

    



main()
