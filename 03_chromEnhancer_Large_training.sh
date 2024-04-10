WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/data/LARGE_NETWORK
HEADER="#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=12:00:00

module load anaconda3_gpu/23.9.0
cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir \$tdir
cp -r $DATADIR \$tdir/
"

TRAIN=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/train_network.py

CELLLINE=("K562")
CHROM=("chr12-chr8")

LOGS=$WORKINGDIR/logs-large
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-large
mkdir -p $SLURM
OUTPUT=$WORKINGDIR/output-large
mkdir -p $OUTPUT

for CELL in ${CELLLINE[@]}; do
        for CHRPAIR in ${CHROM[@]}; do
		echo -e $CELL"\t"$CHRPAIR
		# Cleanup in case of prior run
		rm -f $SLURM/large_$CELL\_$CHRPAIR\-1.slurm
		# Output header
		echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-1.slurm
		echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"1\" > $LOGS/$CELL\_$CHRPAIR\_1log.out &" >> $SLURM/large_$CELL\_$CHRPAIR\-1.slurm

#		sbatch $SLURM/parameter_$CELL\_$CHRPAIR\-1.slurm
	done

done
