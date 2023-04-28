#!/bin/bash

# Get CUDA version from nvidia-smi
cuda_version=$(/usr/bin/nvidia-smi --query-gpu=cuda_version --format=csv,noheader | cut -d. -f1-2)

# Download and install Anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh -b 
rm Anaconda3-2020.07-Linux-x86_64.sh

# Update PATH variable
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
. ~/.bashrc
sleep 5

# Initialize conda
conda init
sleep 5
# Update conda and install packages
conda update -n base -c defaults conda -y
conda update --all -y
sleep 5
conda env update -f environment.yml

