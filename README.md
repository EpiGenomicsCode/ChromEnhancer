# Adversarial attack of sequence-free enhancer prediction identifies patterns of chromatin architecture

## Jamil Gafur<sup>1</sup>, Olivia W Lang<sup>2</sup>, & William KM Lai<sup>2,3</sup>

<sup>1</sup> Department of Computer Science, University of Iowa, Iowa City, Iowa 52242, USA

<sup>2</sup> Department of Molecular Biology and Genetics, Cornell University, Ithaca, NY 14850, USA

<sup>3</sup> Department of Computational Biology, Cornell University, Ithaca, NY 14850, USA

## Abstract

The wide range of cellular complexity created by multicellular organisms is due in large part to the intricate and synergistic interplay of regulatory complexes throughout the eukaryotic genome. These regulatory elements ‘enhance’ specific gene programs and have been shown to operate in diverse networks that are distinct across cell states of the same organism. Attempts to characterize and predict enhancers have typically focused on leveraging information-dense DNA sequence in parallel with epigenomic assays. We examined the viability of enhancer prediction using only a minimal set of epigenomic datasets without direct DNA information. We demonstrate that chromatin datasets are sufficient to identify enhancers genome-wide with high accuracy. By training networks leveraging data from multiple cell types simultaneously, we generated a cell-type invariant enhancer prediction platform that utilized only the patterns of protein binding for inference. We also showed the utility of swarm-based adversarial attacks (APSO) to deconvolute trained genomic neural networks for the first time. Critically, unlike saliency mapping or other game-theory based approaches, APSO is completely network-architecture independent and can be applied to any prediction engine to derive the features that drive inference.

## Network Training and Analysis

This repo has been designed to completely re-generate the analysis and figures contained within the manuscript (DOI: XXXXXX). Random seeds have been set as neccessary to maximize reproducibility. Scripts should be executed in numerical order per sub-folder.

### 1. Preprocessing

* Navigate to the Preprocessing folder and follow the instructions

### 2. Building the environment

```
conda create --prefix ~/work/ChromEnh python=3.9
```

```
conda activate ~/work/ChromEnh
```

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Validate pytorch is able to see GPU:
```
python
import torch
torch.cuda.is_available()

```

```
pip install h5py tqdm matplotlib scikit-learn seaborn

```

The APSO code should be cloned from here:
```
git clone https://github.com/EpiGenomicsCode/Adversarial_Observation
```

And moved into the 'util' folder.

### 3. Download data

Shell scripts should be executed sequentially in:
 - 00_preprocessing

### 4. Train networks

There are currently three different studies that can be performed:

1. Chromosome dropout traininig
2. Cell line independent training
3. Large model training

These networks can be executed asynchronously in:
 - 01_network_training

### 5. Explainable AI analysis

Shell scripts should be executed sequentially in:
 - 03_xai

### 6. Large network analysis

Shell scripts should be executed sequentially in:
 - 04_largenetwork


### FAQ

1. How do I add my own model?
* Navigate to the util/Model folder and add your model as a new python file
* Edit the loadModel function in the util.py file
