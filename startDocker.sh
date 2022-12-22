nohups docker run --rm --gpus all --name cge -v $PWD:/work cornell_genetics /bin/sh -c 'cd /work; sudo rm -rf output; python runHomoModels.py' &
