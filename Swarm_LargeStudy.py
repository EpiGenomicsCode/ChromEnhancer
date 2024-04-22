import torch
import os
import glob
import numpy as np
import tqdm
from util.Adversarial_Observation.Swarm_Observer.Swarm import PSO
from util.Adversarial_Observation.Adversarial_Observation.utils import seedEverything
from util.Adversarial_Observation.Adversarial_Observation.Attacks import *
from util.models.ChrNet1 import Chromatin_Network1
from util.models.ChrNet4 import Chromatin_Network4
import matplotlib.pyplot as plt

def costFunc(model, input):
    """
    Args:
        model (torch.nn.Module): Pre-trained PyTorch model.
        input (numpy.ndarray): Input tensor.
    
    Returns:
        float: Output value of the model for the input.
    """
    val = model(torch.tensor(input).to(torch.float32)).item()
    return val

def getPositions(APSO):
    """
    Extracts and normalizes the positions from the given APSO swarm.
    
    Args:
        APSO (Swarm_Observer.Swarm.PSO): Particle Swarm Optimization object.
    
    Returns:
        list: List of normalized positions.
    """
    positions = []
    for particle in APSO.swarm:
        pos = particle.position_i
        # Normalize the position to be between 0 and 1
        pos = (pos - torch.min(pos)) / (torch.max(pos) - torch.min(pos))
        particle.position_i = pos
        positions.append(pos)
    return positions

def visualize(positions, epoch, name, model):
    """
    Saves the positions and activations as npy files and then plots the position and activation. Then condenses the activation.
    
    Args:
        positions (list): List of positions.
        epoch (int): Current epoch number.
        name (str): Name of the output directory.
        model (torch.nn.Module): Pre-trained PyTorch model.
    """
    # Get the cost of each position
    cost = np.array([costFunc(model, i) for i in positions])
    print(cost)
    import pdb; pdb.set_trace()
    # Sort the positions by cost
    sorted_indices = np.argsort(cost)
    # Convert tensory to numpy
    tensor_arrays = [pos.numpy() for pos in positions]
    # Resort numpy array by cost sort
    positions = [array for _, array in sorted(zip(sorted_indices, tensor_arrays))]
    # Get the activations
    activations = []
    for i in range(len(positions)):
        activations.append(saliency_map(torch.tensor(positions[i], dtype=torch.float32), model))
    activations = [act.numpy() for act in activations]
    
    # Set figure size to 12, 6
    plt.figure(figsize=(12, 6))
    plt.imshow(positions.reshape(len(positions), -1), cmap="jet")
    plt.colorbar()
    plt.savefig(f"{name}/{epoch}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(activations, cmap="jet")
    plt.colorbar()
    plt.savefig(f"{name}/{epoch}_act.png")
    plt.close()

    # Sum every 100 values
    compressed_activation = []
    for point in activations:
        compress = [np.sum(point[i:i+100]) for i in range(0, len(point), 100)]
        compressed_activation.append(compress)
    compressed_activation = np.array(compressed_activation)

    plt.figure(figsize=(12, 6))
    plt.imshow(compressed_activation.reshape(len(compressed_activation), -1), cmap="jet")
    plt.colorbar()
    plt.savefig(f"{name}/{epoch}_act_compressed.png")
    plt.close()

    compressed_position = []
    for point in positions:
        compress = [np.sum(point[i:i+100]) for i in range(0, len(point), 100)]
        compressed_position.append(compress)
    compressed_position = np.array(compressed_position)

    plt.figure(figsize=(12, 6))
    plt.imshow(compressed_position.reshape(len(compressed_position), -1), cmap="jet")
    plt.colorbar()
    plt.savefig(f"{name}/{epoch}_compressed.png")
    plt.close()

def main():
    points = 250
    initialPoints = []
    epochs = 20
    sparcity = 0.8
    for i in range(points):
        arr = np.random.rand(1, 33000)
        mask = np.random.choice([0, 1], size=arr.shape, p=[0.8, 0.2])
        arr = arr * mask
        initialPoints.append(arr)

#        p = np.random.rand(1, 33000)
#        # Set 80% of the points to 0 randomly
#        p[0, np.random.choice(33000, int(33000 * sparcity), replace=False)] = 0
#        import pdb; pdb.set_trace()
#        initialPoints.append(p)

    files = glob.glob("output-large/modelWeights/*_1_*pt")
    for file in reversed(files):
        name = os.path.splitext(os.path.basename(file))[0]
        os.makedirs(name, exist_ok=True)
        modelType = file.split("_")[1]
        model = Chromatin_Network1("", 33000) if modelType == '1' else Chromatin_Network4("", 33000)
        model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        model.eval()
        APSO = PSO(torch.tensor(initialPoints).reshape(-1, 33000), costFunc, model, w=0.8, c1=0.5, c2=0.5)
        for epoch in tqdm.tqdm(range(1, epochs + 1)):
            visualize(getPositions(APSO), epoch, name, model)
            APSO.step()

if __name__ == "__main__":
    main()