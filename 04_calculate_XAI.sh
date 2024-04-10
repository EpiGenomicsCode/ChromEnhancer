WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/data
HEADER="#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=12:00:00

module load anaconda3_gpu/23.9.0
source activate /scratch/bbse/wklai/EnhancerNN/shap
cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir -p \$tdir
cp -r $DATADIR \$tdir/
"

CALC=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/calculate_SHAP-Gradient.py

# Cleanup in case of prior run
rm -f xai_CLD-*
# Set model
MODELPATH=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/output-CELL/modelWeights/CLD_study_-_test_-_valid_-_model4_clkeep_HepG2-MCF7-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII_type-1_epoch_19.pt
MODELID=4
slurmID=xai_CLD-0_model$MODEL\.sh
echo "$HEADER" > $slurmID
echo "python $CALC --fileInput="\$tdir/data/CELL_NETWORK/" --modelPath $MODELPATH --modelID $MODELID --cellLine HepG2 MCF7 A549" >> $slurmID

# for file in *CLD*slurm; do
# 	sbatch $file;
# done
