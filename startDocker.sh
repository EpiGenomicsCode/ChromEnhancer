#!/bin/bash

OUT=$(hostname).tar.gz

nohup docker run --rm --gpus all --name cge -e "HOSTNAME=$(cat /etc/hostname)" -v $PWD:/work cornell_genetics /bin/sh -c 'cd /work;  rm -rf output; rm *gz; python runHomoModels.py; tar -czvf $OUT output/' & 