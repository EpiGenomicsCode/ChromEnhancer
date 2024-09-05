import torch
import os
import argparse
import numpy as np
import tqdm
from util.Adversarial_Observation.Swarm_Observer.Swarm import PSO
from util.Adversarial_Observation.Adversarial_Observation.utils import seedEverything
from util.Adversarial_Observation.Adversarial_Observation.Attacks import *

from captum.attr import IntegratedGradients, Saliency

from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5, ChrNet6
import matplotlib.pyplot as plt

from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the study')
    parser.add_argument('--modelPath', type=str, required=True, help='Location of trained model')
    parser.add_argument('--modelSize', type=int, required=True, help='Model size')
    parser.add_argument('--modelType', type=int, required=True, help='Model number')
    parser.add_argument('--outputPath', type=str, default="./output/", help='Output file path')
    parser.add_argument('--particleNum', type=int, default=500, help='Number of particles in swarm')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--sparsity', type=float, default=0.999, help='Sparsity of initilization vectors (float 0-1)')
    parser.add_argument('--inertiaWeight', type=float, default=0.8, help='How much from the previous epoch are we interested in retaining for future epochs (float 0-1)')
    parser.add_argument('--cognitiveWeight', type=float, default=0.2, help='How much should we focus on going to each particles previous best (float 0-1)')
    parser.add_argument('--socialWeight', type=float, default=0.2, help='How much should we focus on going to the best found particle (float 0-1)')
    parser.add_argument('--randomSeed', type=int, help='Set random seed for reproducibility (default off, otherwise receives int)')
    return parser.parse_args()

def costFunc(model, input):
    """
    Args:
        model (torch.nn.Module): Pre-trained PyTorch model.
        input (numpy.ndarray): Input tensor.
    
    Returns:
        float: Output value of the model for the input.
    """
    val = 1 - model(torch.tensor(input).to(torch.float32)).item()
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

def visualize(APSO, gradients, saliency, epoch, name, model, output_path):
    """
    Saves the positions and activations as npy files and then plots the position and activation. Then condenses the activation.
    
    Args:
        positions (list): List of positions.
        epoch (int): Current epoch number.
        name (str): Name of the output directory.
        model (torch.nn.Module): Pre-trained PyTorch model.
    """
    # Get the cost of each position
    cost = np.array([costFunc(model, particle.position_i) for particle in APSO.swarm])
    #print(f"cost: {cost}")
    # Sort the positions by cost
    sorted_indices = np.argsort(cost)
    # Convert tensor to numpy
    tensor_arrays = [particle.position_i.numpy() for particle in APSO.swarm]
    # Resort numpy array by cost sort
    positions = [array for _, array in sorted(zip(sorted_indices, tensor_arrays))]

    # Zero-padding the epoch number
    epoch_padded = f"{epoch:02d}"

    ig_attr_test = gradients.attribute(torch.tensor(np.array(positions)).float(), n_steps=50)
    #print(ig_attr_test)
    # Sum every 100 values
    compressed_activation = []
    for point in ig_attr_test:
        rowAPSO = np.array(point)
        compress = [np.sum(rowAPSO[i:i+100]) for i in range(0, len(rowAPSO), 100)]
        compressed_activation.append(compress)
    compressed_activation = np.array(compressed_activation)

    plt.figure(figsize=(6, 6))
    plt.imshow(compressed_activation.reshape(len(compressed_activation), -1), cmap="seismic", aspect='auto', vmin=-0.5, vmax=0.5, interpolation='none')
    plt.colorbar()
    plt.title(f"Compressed Integrated Gradient at Epoch {epoch}")
    if epoch > 19:
        plt.savefig(f"{output_path}/{epoch_padded}_{name}_grad_compressed.svg", format='svg')
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_grad_compressed.png", format='png')
    plt.close()

    sal_attr_test = saliency.attribute(torch.tensor(np.array(positions)).float())
   # Sum every 100 values
    compressed_saliency = []
    for point in sal_attr_test:
        rowAPSO = np.array(point)
        compress = [np.sum(rowAPSO[i:i+100]) for i in range(0, len(rowAPSO), 100)]
        compressed_saliency.append(compress)
    compressed_saliency = np.array(compressed_saliency)

    plt.figure(figsize=(6, 6))
    plt.imshow(compressed_saliency.reshape(len(compressed_saliency), -1), cmap="seismic", aspect='auto', interpolation='none')
    plt.colorbar()
    plt.title(f"Compressed Saliency at Epoch {epoch}")
    if epoch > 19:
        plt.savefig(f"{output_path}/{epoch_padded}_{name}_saliency_compressed.svg", format='svg')
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_saliency_compressed.png", format='png')
    plt.close()

    compressed_position = []
    for point in positions:
        compress = [np.sum(point[i:i+100]) for i in range(0, len(point), 100)]
        compressed_position.append(compress)
    compressed_position = np.array(compressed_position)

    plt.figure(figsize=(6, 6))
    plt.imshow(compressed_position.reshape(len(compressed_position), -1), cmap="PRGn", aspect='auto', interpolation='none')
    plt.colorbar()
    plt.title(f"Compressed Position at Epoch {epoch}")
    if epoch > 19:
        plt.savefig(f"{output_path}/{epoch_padded}_{name}_particle_compressed.svg", format='svg')
    plt.savefig(f"{output_path}/{epoch_padded}_{name}_particle_compressed.png", format='png')
    plt.close()

    if epoch > 19:
        # Output current swarm positions to TSV
        np.savetxt(f"{output_path}/{epoch_padded}_{name}_particle_complessed.tsv", compressed_position, delimiter='\t')
        # Output current swarm activation to TSV
        np.savetxt(f"{output_path}/{epoch_padded}_{name}_pso_act.tsv", compressed_activation, delimiter='\t')
        # Output current swarm saliency to TSV
        np.savetxt(f"{output_path}/{epoch_padded}_{name}_pso_sal.tsv", compressed_saliency, delimiter='\t')

def create_gif(input_folder, output_path, file_type, duration):
    # Get all PNG files from the input folder
    images = [img for img in os.listdir(input_folder) if img.endswith(file_type)]
    images.sort()  # Sort images by name

    # Ensure there are images to process
    if not images:
        raise ValueError("No PNG images found in the specified folder")

    # Create a list of image objects
    frames = [Image.open(os.path.join(input_folder, img)) for img in images]

    # Save as a GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration)

    # Delete all PNG files after the GIF is created
    for img in images:
        os.remove(os.path.join(input_folder, img))

def load_model(model_number, name="", input_size=500):
    model_classes = {
        1: ChrNet1.Chromatin_Network1,
        2: ChrNet2.Chromatin_Network2,
        3: ChrNet3.Chromatin_Network3,
        4: ChrNet4.Chromatin_Network4,
        5: ChrNet5.Chromatin_Network5,
        6: ChrNet6.Chromatin_Network6
    }
    if model_number in model_classes:
        return model_classes[model_number](name, input_size)
    else:
        raise ValueError(f"Invalid model number {model_number}")

def main():
    args = parse_arguments()

    if args.randomSeed is not None:
        seedEverything(args.randomSeed)

    points = args.particleNum
    epochs = args.epochs
    sparsity =  args.sparsity

    modelSize = args.modelSize
    sparsity = args.sparsity
    # Model 1
    #sparsity = 0.999
    # Model 6
    #sparsity = 0.9999

    # Make output directory if doesn't exist
    os.makedirs(args.outputPath, exist_ok=True)

    print(f"starting run with {epochs} epochs, {points} points and a sparsity of {sparsity}")

    # Intialize the random matrix
    initialPoints = []
    for i in range(points):
        arr = np.random.rand(1, modelSize)
        mask = np.random.choice([0, 1], size=arr.shape, p=[sparsity, 1-sparsity])
        arr = arr * mask
        initialPoints.append(arr)

    # Load the trained model
    model = load_model(args.modelType, "", modelSize)
    model.load_state_dict(torch.load(args.modelPath, map_location=torch.device('cpu')))
    model.eval()

    # Initialize integrated gradient
    integrated_gradients = IntegratedGradients(model)
    saliency =  Saliency(model)

    # Get the file name
    name = os.path.splitext(os.path.basename(args.modelPath))[0]

    #Initialize the APSO swarm
    # Model 1
    APSO = PSO(torch.tensor(initialPoints).reshape(-1, modelSize), costFunc, model, w=0.005, c1=0.005, c2=0.2)
    # Model 6
    #APSO = PSO(torch.tensor(initialPoints).reshape(-1, 33000), costFunc, model, w=0.005, c1=0.2, c2=0.2)

    # Run the swarm, outputing the matrix for every epoch
    visualize(APSO, integrated_gradients, saliency, 0, name, model, args.outputPath)
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        APSO.step()
        visualize(APSO, integrated_gradients, saliency, epoch, name, model, args.outputPath)

    #Output the final GIF of the swarm
    input_folder = f"{args.outputPath}"
    duration = 500  # Duration between frames in milliseconds
    output_path = f"{args.outputPath}/{name}_swarm_pos.gif"
    create_gif(input_folder, output_path, f"{name}_particle_compressed.png", duration)
    output_path = f"{args.outputPath}/{name}_swarm_act.gif"
    create_gif(input_folder, output_path, f"{name}_grad_compressed.png", duration)
    output_path = f"{args.outputPath}/{name}_swarm_sal.gif"
    create_gif(input_folder, output_path, f"{name}_saliency_compressed.png", duration)

if __name__ == "__main__":
    main()
