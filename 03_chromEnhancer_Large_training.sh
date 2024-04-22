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
#SBATCH --time=48:00:00

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
		echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"1\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_1log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-1.slurm

                # Output header
                echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-2.slurm
                echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"2\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_2log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-2.slurm

                # Output header
                echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-3.slurm
                echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"3\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_3log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-3.slurm

                # Output header
                echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-4.slurm
                echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"4\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_4log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-4.slurm

                # Output header
                echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-5.slurm
                echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"5\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_5log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-5.slurm

                # Output header
                echo "$HEADER" > $SLURM/large_$CELL\_$CHRPAIR\-6.slurm
                echo "python $TRAIN --fileInput="\$tdir/LARGE_NETWORK/" --fileOutput=$OUTPUT --parameterLDS --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"6\" --epochs 200 --batch_size 512 > $LOGS/$CELL\_$CHRPAIR\_6log.out" >> $SLURM/large_$CELL\_$CHRPAIR\-6.slurm

	done
done

cd $SLURM
for CELL in ${CELLLINE[@]}; do
        for CHRPAIR in ${CHROM[@]}; do
                sbatch large_$CELL\_$CHRPAIR\-1.slurm
                sbatch large_$CELL\_$CHRPAIR\-2.slurm
                sbatch large_$CELL\_$CHRPAIR\-3.slurm
                sbatch large_$CELL\_$CHRPAIR\-4.slurm
                sbatch large_$CELL\_$CHRPAIR\-5.slurm
                sbatch large_$CELL\_$CHRPAIR\-6.slurm
	done
done
