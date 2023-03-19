from util.model import *
import torch
from util.Adversarial_Research.Adversarial_Observation.utils import *
from util.Adversarial_Research.Adversarial_Observation.visualize import *
from util.Adversarial_Research.Adversarial_Observation.Swarm_Observer.Swarm import PSO
from util.Adversarial_Research.Adversarial_Observation.Attacks import *
import os
import tqdm

import pandas as pd
import numpy as np
import imageio

def costFunc(model, input):
    return  model(input).item()

def main():
    # load the model
    model = Chromatin_Network4("test", 500)
    # load the weights
    model.load_state_dict(torch.load("id_A549_study_chr10-chr17_model_4_train_chr17_test_chr10_type_-2.pt"))
    model.eval()

    attack_model_apso_random(model, (1, 500), 50, 20, "./Swarm_Output")

def attack_model_apso_random(model: torch.nn.Module, 
                             input_shape: Tuple, 
                             num_of_inputs: int, 
                             epochs: int, 
                             outdir: str = "./Output/") -> None:
    """
    This function will take a model and will randomly generate inputs to the model and attack those inputs
    :param model: the model you want to attack
    :param input_shape: the shape of the inputs to be generated
    :param num_of_inputs: the number of inputs to be generated
    :param epochs: the number of iterations for the attack
    """

    # create the output directory
    os.makedirs('{}'.format(outdir), exist_ok=True)

    # Step 5: Generate the inputs to be attacked
    adversarial_inputs = generate_random_inputs(num_of_inputs, input_shape)


    # Step 6: Attack the model
    attackLoop(adversarial_inputs, model, epochs, outdir)

def attackLoop(inputData: np.ndarray, model, epochs: int, outdir: str, pca = None) -> None:
    """
    This function runs an APSO attack on a model.
    Parameters:
    inputData (ndarray): The input data for the attack.
    model: The target model to attack.
    epochs (int): The number of epochs to run the attack for.
    outdir (str): The output directory for logging data.
    pca (sklearn.decomposition.PCA): The PCA data to plot the swarm in PCA space. Defaults to None.
    Returns:
    None
    """
    # Initialize the swarm
    # set every input in inputData to have a sparcity of .4
    sparse = .9

    for i in range(inputData.shape[0]):
        numzero = int(inputData[i].shape[1] * sparse)
        # get a mask containing numzero 0s
        zeroindex = np.random.choice(inputData[i].shape[1], int(numzero), replace=False)
        # set the 0s
        inputData[i][0][zeroindex] = 0
    
    apso = PSO(torch.tensor(inputData).to(torch.float32), costFunc, model, w=.1, c1=.5, c2=.5)
    

    # Fancy progress bar
    loop = tqdm.tqdm(range(1, epochs+1))
    plotimg(apso, 0, model, outdir)
    # Loop through the attack
    for i in loop:
        # Step the swarm
        positions, best_positions = apso.step()
        # #==================Logging/Visualization===============================
        #     # Get the positions to fit the model
        # positions = cleanPositions(positions)

        #     # Get the highest confidence prediction and the predictions of the swarm
        # best_score, newpredictions = bestInSwarm(model, positions, i, endValue,outdir)

        #     # Sum the positions of the swarm and get its confidence prediction
        # whole_label = convergeOfSwarm(model, positions, i, outdir)

        #     # calculate the percent of particles the converged to the endValue
        # per = np.sum(newpredictions[:, endValue] > .5)/len(newpredictions)

        # #================End Logging/Visualization===============================
        
        # Update the progress bar
        # loop.set_description(f"APSO best score: {best_score:.2f}, Sum predicted label: {whole_label}, Percent of particles labeled as {endValue}: {per:.2f}")
        plotimg( apso, i, model, outdir)
        # create a gif
        imageio.mimsave("img.gif", [imageio.imread("{}/img_{}.png".format(outdir, j)) for j in range(0, i)], fps=.5)
    apso.save_history("temp.csv")


def plotimg(apso, i, model, directory):
    img = []
    value = []
    salency = []
    binData = []
    for datarow in [particle.position_i for particle in apso.swarm]:
        salency_out = saliency_map(datarow, model)
        inputData = datarow.numpy()
        # normalize the data
        inputData = inputData.flatten()

        outputData = model(torch.tensor(inputData).to(torch.float32)).item()
        bin = np.clip(np.array([np.mean(inputData[i:i+100]) for i in range(0, len(inputData), 100) ]),0,1)
        # clip the data between 0 and 1
        inputData = np.clip(inputData, 0, 1)
        # inputData = (inputData - np.min(inputData)) / (np.max(inputData) - np.min(inputData))
        img.append(inputData)
        value.append(outputData)
        salency.append(salency_out)
        binData.append(bin)



    # sort ascending

    sorted_paird = sorted(zip(salency,img, value), key=lambda x: x[2], reverse=True)
    salency = [i[0] for i in sorted_paird]
    img = [i[1] for i in sorted_paird]
    value = [i[2] for i in sorted_paird]
    # plot 2 subplots
    # font size
    plt.rcParams.update({'font.size': 30})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig size
    fig.set_size_inches(40, 15)
    # plot the salency map
    
    ax1.imshow(salency, cmap="jet", aspect="auto")
    ax1.set_title("Salency Map")
    # ax2 colorbar
    fig.colorbar(ax1.get_images()[0], ax=ax1)
    # plot the input data
    ax2.set_title("Input Data")
    # plot the input data
    ax2.imshow(img, cmap="jet", aspect="auto")
    fig.colorbar(ax2.get_images()[0], ax=ax2)

    x =["CTCF", "H3k4me", "H3k27ac", "P300", "PolII"]
    spacing = np.round(np.linspace(50, 450, len(x)))
    ax2.set_xticks(spacing, x)    
    spacing = np.round(np.linspace(0, len(value)-1, 10))
    # get the value at the index of the rounded array
    ticks = [round(value[int(i)],4) for i in spacing]
    # plot the y axis as the value array
    ax2.set_yticks(spacing, ticks)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Feature")
    ax2.set_title("best:{}, epoch:{}".format(round(max(value),5), i))
    
    ax3.set_title("Bin Data")
    ax3.imshow(binData, cmap="jet", aspect="auto")
    fig.colorbar(ax3.get_images()[0], ax=ax3)
    spacing = np.round(np.linspace(0, 4, len(x)))
    ax3.set_xticks(spacing, x)  
    # spacing = np.round(np.linspace(0, len(value)-1, 10))
    # get the value at the index of the rounded array
    # ticks = [round(value[int(i)],4) for i in spacing]
    # plot the y axis as the value array
    # ax3.set_yticks(spacing, ticks)
    # ax3.set_ylabel("Confidence")
    ax3.set_xlabel("Feature")
    ax3.set_title("best:{}, epoch:{}".format(round(max(value),5), i))
    
    
    # set ax1 and ax2 to tight layout
    fig.tight_layout()
    plt.savefig("{}/img_{}.png".format(directory, i))
    plt.close()
    plt.clf()


if __name__ == "__main__":
    main() 