#!/bin/bash

wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
sh Anaconda3-2020.07-Linux-x86_64.sh -b -p /home/exouser/anaconda3
rm Anaconda3-2020.07-Linux-x86_64*
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc
. ~/.bashrc
conda init
conda update conda -y
conda update anaconda -y
conda update --all -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
