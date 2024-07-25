import sys, os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from captum.attr import IntegratedGradients, GradientShap, DeepLiftShap, Saliency

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.models import ChrNet1, ChrNet2, ChrNet3, ChrNet4, ChrNet5, ChrNet6
from util import dataset

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the study')
    parser.add_argument('--fileInput', type=str, default="./data/CELL_NETWORK/", help='Location of training and test data')
    parser.add_argument('--outputPath', type=str, default="./output/", help='Output file path')
    parser.add_argument('--modelPath', type=str, required=True, help='Location of trained model')
    parser.add_argument('--modelType', type=int, required=True, help='Model number')
    parser.add_argument('--dataType', type=str, default="-1", help='Data replicate ID')
    parser.add_argument('--cellLine', nargs='+', default=["A549", "MCF7", "HepG2", "K562"], help='Run the study on the cellLine')
    parser.add_argument('--chromData', nargs='+', default=["CTCF", "H3K4me3", "H3K27ac", "p300", "PolII", "H3K36me3", "H3K27me3", "H3K4me1"], help='Run the study using the following chromatin datasets')
    parser.add_argument('--batch_size', type=int, default=2048, help='Run the study on the batch size')
    parser.add_argument('--bin_size', type=int, default=65536, help='How many bins to use when loading the data')
    return parser.parse_args()

def calc_xai(model, model_name, output_path, cell_use, cell_hold, chr_use, file_location, data_type, batch_size=2048):
    train, test, valid = dataset.getData(trainLabel="", testLabel="", validLabel="", chrUse=chr_use, cellUse=cell_use, cellHold=cell_hold, bin_size=batch_size, fileLocation=file_location, dataTypes=data_type)
    x_train, y_train = next(iter(DataLoader(train, batch_size=batch_size, shuffle=True)))

    x_test, y_test = [], []
    x_null, y_null = [], []
    print("Loading test data...")
    for i, (data, label) in enumerate(tqdm(test, desc="Processing test data")):
        if label[0] > 0.9:
            x_test.append(data)
            y_test.append(label[0])
        else:
            x_null.append(data)
            y_null.append(label[0])
#        if len(x_test) >= 2000:  # Only take first N samples
#            break

    print(f"Total samples assessed: {i+1}")
    print(f"Samples passing threshold: {len(x_test)}")

    x_test, y_test = torch.tensor(np.array(x_test)), torch.tensor(np.array(y_test).reshape(-1, 1))
    test = TensorDataset(x_test, y_test)
    model, x_train, x_test, y_train = model.cpu(), x_train.cpu(), x_test.cpu(), y_train.cpu()

    x_null = torch.tensor(np.array(x_null))

    print("Initializing Captum")
    saliency =  Saliency(model)
    integrated_gradients = IntegratedGradients(model)
    gradient_shap = GradientShap(model)
    deeplift_shap = DeepLiftShap(model)

    print("Calculating Saliency for TEST data")
    sal_attr_test = saliency.attribute(x_test.float())

    print("Calculating Integrated Gradients for TEST data")
    ig_attr_test = integrated_gradients.attribute(x_test.float(), internal_batch_size=2000, n_steps=50)
    
#    print("Calculating Gradient SHAP for TEST data")
#    gs_attr_test = gradient_shap.attribute(x_test.float(), n_samples=5, stdevs=0.0001, baselines=x_null.float())
    print("Calculating Deeplift SHAP for TEST data")
    gs_attr_test = gradient_shap.attribute(x_test.float(), baselines=x_null.float())

    np.savetxt(os.path.join(output_path, f"{model_name}_xai_orig.tsv"), x_test.numpy(), delimiter='\t')
    np.savetxt(os.path.join(output_path, f"{model_name}_xai_saliency.tsv"), sal_attr_test.numpy(), delimiter='\t')
    np.savetxt(os.path.join(output_path, f"{model_name}_xai_gradient.tsv"), ig_attr_test.numpy(), delimiter='\t')
    np.savetxt(os.path.join(output_path, f"{model_name}_xai_shap.tsv"), gs_attr_test.numpy(), delimiter='\t')

    print("XAI calculation complete")

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

    cell_use = args.cellLine
    cell_holdout = list(set(["A549", "MCF7", "HepG2", "K562"]) - set(cell_use))
    chrom_data = args.chromData

    print(f"Loading model: {args.modelPath}")
    model = load_model(args.modelType, "", 800)
    model.load_state_dict(torch.load(args.modelPath, map_location='cpu'))

    calc_xai(model, os.path.basename(args.modelPath), args.outputPath, cell_use, cell_holdout, chrom_data, args.fileInput, args.dataType, args.batch_size)

if __name__ == "__main__":
    main()
