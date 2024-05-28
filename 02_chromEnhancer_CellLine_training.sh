WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen
DATADIR=$WORKINGDIR/data/CELL_NETWORK
# HEADER="#!/bin/bash
# #SBATCH -A bbse-delta-gpu
# #SBATCH --partition=gpuA100x4
# #SBATCH --gpus=1
# #SBATCH --nodes=1
# #SBATCH --tasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=48g
# #SBATCH --time=12:00:00
# 
# module load anaconda3_gpu/23.9.0
# cd $WORKINGDIR
# tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
# mkdir -p \$tdir
# cp -r $DATADIR \$tdir/
# "

HEADER="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=60GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --account=gja2_a_gpu
#SBATCH -p sla-prio,burst
#SBATCH -q burst4x
#SBATCH --exclude=p-gc-3024

module load anaconda
conda activate /storage/home/wkl2/work/ChromEnh

cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir \$tdir
cp -r $DATADIR \$tdir/
"

TRAIN=$WORKINGDIR/train_network.py

LOGS=$WORKINGDIR/logs-cell
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-cell
mkdir -p $SLURM
OUTPUT=$WORKINGDIR/output-cell
mkdir -p $OUTPUT

# Cleanup in case of prior run
rm -f $SLURM/parameter_CLD-*

# Set model
MODEL=1
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID

# Set model
MODEL=2
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID

# Set model
MODEL=3
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID

# Set model
MODEL=4
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID


# Set model
MODEL=5
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID

# Set model
MODEL=6
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > $LOGS/CLD-0_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > $LOGS/CLD-1_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > $LOGS/CLD-2_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_0log_model$MODEL\.out &" >> $SLURM/$slurmID
echo "sleep 30" >> $SLURM/$slurmID
echo "python $TRAIN --fileInput="\$tdir/CELL_NETWORK/" --fileOutput=$OUTPUT --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > $LOGS/CLD-3_1log_model$MODEL\.out" >> $SLURM/$slurmID
echo "wait" >> $SLURM/$slurmID

cd $SLURM
for file in *slurm; do
	sbatch $file;
done
