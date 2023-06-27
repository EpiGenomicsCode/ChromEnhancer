from util.models import ChrNet4
from util.dataset import Chromatin_Dataset
import shap
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import Adversarial_Observation as AO

def bin_array(input):
    output = []
    for i in range(0, len(input), 100):
        output.append(np.mean(input[i:i+100]))
    return output


# Load the model
model = ChrNet4.Chromatin_Network4("", 32900)
model.load_state_dict(torch.load("/home/exouser/Enhanced_Predictions_For_Enhancers/MOUNT/DATA/output/LargeDataset1_4_epoch_49.pt"))
model.train()
# Create the dataset
dataset = Chromatin_Dataset(cellLine="K562",
                            chrUse=[],
                            dataTypes="",
                            label="chr12-chr8",
                            fileLocation="./Data/230415_LargeData/TRAIN/",
                            chunk_size=32,
                            mode="train")

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get the data
x_train, y_train = next(iter(dataloader))

background = x_train[np.random.choice(x_train.shape[0], 10, replace=False)]
background = background.to(torch.float32)

e = shap.DeepExplainer(model, background)

data_index = 300

orig_list = []
shap_list = []
actv_list =[]

shap_values = e.shap_values(x_train[:data_index].to(torch.float32))

assert len(shap_values) == len(x_train[:data_index])
for index in range(len(shap_values)):
    print(f"Plotting {index}")
    print(shap_values[index].shape)
    print(x_train[index].shape)
    orig_list.append(x_train[index].detach().numpy())
    shap_list.append(shap_values[index])
    model.train()
    actv_list.append(AO.Attacks.gradient_map(orig_list[0].reshape(1,1,32900), model, (1,1,32900))[0])

    binned_shap = bin_array(shap_values[index])
    binned_data = bin_array(x_train[index].detach().numpy())
    # stack 50 binned_shap and then 50 binned_data
    stacked_shap = np.stack([binned_shap for _ in range(50)], axis=0)
    stacked_data = np.stack([binned_data for _ in range(50)], axis=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confidence = model(x_train[index].unsqueeze(0).to(torch.float32).to(device)).item()
    fig, ax = plt.subplots(2, 1)

    #  normalize the data between 0 and 1
    stacked_shap = (stacked_shap - np.min(stacked_shap)) / (np.max(stacked_shap) - np.min(stacked_shap))
    stacked_data = (stacked_data - np.min(stacked_data)) / (np.max(stacked_data) - np.min(stacked_data))
    ax[0].imshow(stacked_shap, cmap="jet")
    ax[0].set_title("SHAP")
    ax[1].imshow(stacked_data, cmap="jet")
    ax[1].set_title("Data: " + str(confidence))
    ax[0].set_yticks([])
    ax[1].set_yticks([])

    # colorbar for both axes scale it to the image
    fig.colorbar(ax[0].imshow(stacked_shap, cmap="jet"), ax=ax[0])
    fig.colorbar(ax[1].imshow(stacked_data, cmap="jet"), ax=ax[1])

    plt.savefig(f"shap_{index}.png")
    print("Saved")
    plt.close()

df = pd.DataFrame(shap_list)
# save as csv with no index and no header   
df.to_csv("shap.csv", index=False, header=False)
df = pd.DataFrame(orig_list)
df.to_csv("orig.csv", index=False, header=False)
df = pd.DataFrame(actv_list)
df.to_csv("actv.csv", index=False, header=False)
