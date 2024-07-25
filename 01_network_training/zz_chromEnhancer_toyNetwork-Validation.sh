#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=12:00:00

module load anaconda
conda activate /storage/home/wkl2/work/ChromEnh

INPUT=/storage/group/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen
OUTPUT=/storage/group/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen
TRAIN=/storage/group/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen/train_network.py

cd $INPUT
tdir=/tmp/foo.vElBKSyN6
mkdir $tdir
cp -r $INPUT/data/TOY_NETWORK $tdir/

python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="1" --index="-1"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="2" --index="-1"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="3" --index="-1"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="4" --index="-1"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="5" --index="-1"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="6" --index="-1"

python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="1" --index="-2"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="2" --index="-2"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="3" --index="-2"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="4" --index="-2"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="5" --index="-2"
python $TRAIN --fileInput=$tdir/TOY_NETWORK/ --fileOutput=$OUTPUT/output-chr --parameterCHR --cellLine="K562" --chrPair="chr10-chr17" --model="6" --index="-2"

