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
from itertools import permutations
import pickle


# TODO turn off interpolation for the images
# Not bio ....
# dont do all of the CL only the dropout one

def bin_array(input):
    output = []
    for i in range(0, len(input), 100):
        output.append(np.mean(input[i:i+100]))
    return output

def getDataHighest(cellLines):
    train,test, valid = dataset.getData(cellLineUse=cellLines, chrUse=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"])
    #  go through all test and only use the ones that have high confidence
    x_test = []
    y_test = []
    for data, label in test:
        if label[0] > 0.9:
            x_test.append(data)
            y_test.append(label[0])

    x_test = np.array(x_test)
    y_test = np.array([[i] for i in y_test])
    test = torch.utils.data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    return train, test, valid

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
            modelarchitecture = ChrNet1.Chromatin_Network1("", 500)
            modelarchitecture.load_state_dict(torch.load(model))
        elif "model2" in model:
            modelarchitecture = ChrNet2.Chromatin_Network2("", 500)
            modelarchitecture.load_state_dict(torch.load(model))
        elif "model3" in model:
            modelarchitecture = ChrNet3.Chromatin_Network3("", 500)
            modelarchitecture.load_state_dict(torch.load(model))
        elif "model4" in model:
            modelarchitecture = ChrNet4.Chromatin_Network4("", 500)
            missing = [i for i in["A549", "MCF7", "HepG2", "K562"] if i not in model_name]
            modelarchitecture.load_state_dict(torch.load(model))
            train, test, valid = dataset.getData(cellLineUse=missing, chrUse=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII"])
            # cast valid to a dataloader 
            valid = DataLoader(valid, batch_size=64, shuffle=False)
            
            data = []
            modelarchitecture.eval()
            
            with torch.no_grad():
                for batch in valid:
                    prediction =  modelarchitecture(batch[0].to(torch.float32))
                    data.extend(zip(batch[0].to(torch.float32),prediction))
                    
            os.makedirs(f"./output/valid_predictions/{model_name}/", exist_ok=True)
            # save the predictions as a txt file
            with open(f"./output/valid_predictions/{model_name}/data_{''.join(missing)}.txt", "w") as f:
                for batch, prediction in data:
                    batch = batch.tolist()
                    batch.append(prediction[0].item())
                    f.write(str(batch))
                    f.write("\n")

                
            train, test, valid = getDataHighest(missing)
            Cell_line_independent_model(modelarchitecture, model_name+"_"+''.join(missing), train, test, valid)
        else:
            continue

        # cellline = model_name.split("_")[3]
        # train, test, valid = getData(cellline.split("-"))
        # Cell_line_independent_model(model, model_name, train, test, valid)
        
        

def Cell_line_independent_model(model, model_name, train, test, valid,data_size=100):

    # we are using valid just to make sure the visualizations are correct
    x_train, y_train = next(iter(DataLoader(train, batch_size=data_size, shuffle=True)))
    # x_train, y_train = next(iter(DataLoader(vaild, batch_size=32, shuffle=True)))
    x_test, y_test = next(iter(DataLoader(test, batch_size=data_size, shuffle=True))) # Load the entire test set

    model = model.to('cpu')
    # send everything to cpu
    x_train = x_train.to('cpu')
    x_test = x_test.to('cpu')
    y_test = y_test.to('cpu')
    y_train = y_train.to('cpu')


    background = x_train  # Using the whole training set as background
    background = background.to(torch.float32)
    e = shap.DeepExplainer(model, background)

    # Get SHAP values for the entire test set
    # 
    # TODO: sort the holdout based on highest value from the model predicitons
    # 
    shap_values = e.shap_values(x_test[:data_size].to(torch.float32))

    # Save the original images, labels, and SHAP values in a DataFrame
    data = {
        "Original Image": x_test[:data_size].numpy(),
        "Original Label": y_test[:data_size].numpy(),
        "SHAP Values": shap_values[:data_size]
    }

    # pickle the data
    os.makedirs(f"./output/shap/{model_name}_shap_values", exist_ok=True)
    with open(f"./output/shap/{model_name}_shap_values/data.pkl", "wb") as f:
        pickle.dump(data, f)
    # plot a 3x1 of the original image, the shap values, and the activation map
    for i in range(data_size):
        fig, ax = plt.subplots(3, 1)
        ax[0].imshow(np.tile(x_test[i].numpy().reshape(1, 500), (30, 1)), cmap="jet")
        ax[1].imshow(np.tile(shap_values[i].reshape(1, 500), (30, 1)), cmap="jet")
        gradient_map = AO.Attacks.gradient_map(x_test[i].reshape(1, 1, 500).to(torch.float32), model, (1, 500), backprop_type='guided')
        ax[2].imshow(np.tile(gradient_map.reshape(1, 500), (30, 1)), cmap="jet")
        #  set the titles
        ax[0].set_title(f"Original Image: label {y_test[i].numpy()[0]}")
        ax[1].set_title("SHAP Values")
        ax[2].set_title(f"Activation Map")
        #  remove the axes
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        #  save the figure
        plt.savefig(f"./output/shap/{model_name}_shap_values/{i}.png")
        plt.close()
    

if __name__ == "__main__":
    main()