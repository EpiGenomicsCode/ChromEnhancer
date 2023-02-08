#!/bin/bash
nohup docker run --rm --gpus all --name cge -e "HOSTNAME=$(cat /etc/hostname)" -v $PWD:/work cornell_genetics /bin/sh -c  "cd /work; rm -rf output; rm *tar; python runHomoModels.py; python swarmStudy.py; tar -czvf $(hostname).tar.gz output/" &
