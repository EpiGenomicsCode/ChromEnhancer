#!/bin/bash
# create a string variable for the initializer

initializer="# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/exouser/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/exouser/anaconda/etc/profile.d/conda.sh" ]; then
        . "/home/exouser/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/home/exouser/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<"


wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
sh Anaconda3-2020.07-Linux-x86_64.sh -b -p /home/exouser/anaconda
rm Anaconda3-2020.07-Linux-x86_64*
echo "$initializer" 
echo "$initializer" >> ~/.bashrc

. ~/.bashrc
conda init
conda update conda -y
conda update anaconda -y
conda update --all -y
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
