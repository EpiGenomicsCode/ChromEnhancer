#!/bin/bash
nohups docker run --rm --gpus all --name cge -e "HOSTNAME=$(cat /etc/hostname)" -v $PWD:/work cornell_genetics /bin/sh -c  "cd /work;  python runHomoModels.py; tar -czvf $(hostname).tar.gz output/" &