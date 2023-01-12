#!/bin/bash

OUT=$(hostname).tar.gz
nohup docker run --rm --gpus all --name cge -v $PWD:/work cornell_genetics /bin/sh -c 'cd /work;  rm -rf output; python runHomoModels.py; tar -czvf $OUT output' & 