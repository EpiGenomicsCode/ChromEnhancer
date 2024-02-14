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
cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir -p \$tdir
cp -r $DATADIR \$tdir/
"

TRAIN=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/train_network.py

# Cleanup in case of prior run
rm -f parameter_CLD-*
# Set model
MODEL=1
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID

# Set model
MODEL=2
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID

# Set model
MODEL=3
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID

# Set model
MODEL=4
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID


# Set model
MODEL=5
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID

# Set model
MODEL=6
slurmID=parameter_CLD-0_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 MCF7 > CLD-0_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 MCF7 > CLD-0_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-1_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 HepG2 A549 > CLD-1_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 HepG2 A549 > CLD-1_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-2_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine K562 MCF7 A549 > CLD-2_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine K562 MCF7 A549 > CLD-2_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID
slurmID=parameter_CLD-3_model$MODEL\.slurm
echo "$HEADER" > $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-1\" --cellLine HepG2 MCF7 A549 > CLD-3_0log_model$MODEL\.out &" >> $slurmID
echo "sleep 30" >> $slurmID
echo "python $TRAIN --fileInput="\$tdir/data/CELL_NETWORK/" --parameterCLD --model=\"$MODEL\" --index=\"-2\" --cellLine HepG2 MCF7 A549 > CLD-3_1log_model$MODEL\.out" >> $slurmID
echo "wait" >> $slurmID

for file in *CLD*slurm; do
	sbatch $file;
done
