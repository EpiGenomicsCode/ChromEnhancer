from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4
from util import dataset
import shap
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import Adversarial_Observation as AO
import glob
from tqdm import tqdm
import pickle

def bin_array(input):
    output = []
    for i in range(0, len(input), 100):
        output.append(np.mean(input[i:i+100]))
    return output

def main():
    # Step 1:
    #  1.1 For each model, load the model
    #  1.2 Get the training data and validation data from the dataset
    #  1.3 Create the explainer
    # Step 2:
    #  2.1 For each model, get the shap values for the training data, save it as a csv
    #  2.2 For each model, get the shap values for the validation data, save it as a csv

    cellLine_Independent_model = glob.glob("./output/modelWeights/CLD*19*")
    print(cellLine_Independent_model)
    for model in tqdm(cellLine_Independent_model):
        model_name = model.split("/")[-1][:-3]
        print(model_name)
        if "model1" in model:
            model = ChrNet1.Chromatin_Network1("", 500)
        elif "model2" in model:
            model = ChrNet2.Chromatin_Network2("", 500)
        elif "model3" in model:
            model = ChrNet3.Chromatin_Network3("", 500)
        elif "model4" in model:
            model = ChrNet4.Chromatin_Network4("", 500)
        else:
            continue
        train, test, vaild =  dataset.getData(cellLineUse=["A549", "MCF7", "HepG2", "K562"], chrUse=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"])

        # we are using valid just to make sure the visualizations are correct
            # x_train, y_train = next(iter(DataLoader(train, batch_size=32, shuffle=True)))
        x_train, y_train = next(iter(DataLoader(vaild, batch_size=32, shuffle=True)))
        x_test, y_test = next(iter(DataLoader(test, batch_size=32, shuffle=True))) # Load the entire test set

        background = x_train  # Using the whole training set as background
        background = background.to(torch.float32)

        e = shap.DeepExplainer(model, background)

        # Get SHAP values for the entire test set
        # 
        # TODO: sort the holdout based on highest value from the model predicitons
        # 
        shap_values = e.shap_values(x_test[:32].to(torch.float32))

        # Save the original images, labels, and SHAP values in a DataFrame
        data = {
            "Original Image": x_test[:32].numpy(),
            "Original Label": y_test[:32].numpy(),
            "SHAP Values": shap_values[:32]
        }

        # pickle the data
        os.makedirs(f"./output/shap/{model_name}_shap_values", exist_ok=True)
        with open(f"./output/shap/{model_name}_shap_values/data.pkl", "wb") as f:
            pickle.dump(data, f)

        # plot a 3x1 of the original image, the shap values, and the activation map
        for i in range(32):
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(np.tile(x_test[0].numpy().reshape(1, 500), (30, 1)), cmap="jet")
            ax[1].imshow(np.tile(shap_values[i].reshape(1, 500), (30, 1)), cmap="jet")
            ax[2].imshow(np.tile(AO.Attacks.gradient_map(x_test[0].reshape(1,1, 500).to(torch.float32),model.train(), (1,500))[0], (30, 1)), cmap="jet")
            plt.savefig(f"./output/shap/{model_name}_shap_values/{i}.png")
            plt.close()
        

if __name__ == "__main__":
    main()