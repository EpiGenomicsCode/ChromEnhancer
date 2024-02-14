WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/data
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

CELLLINE=("K562" "HepG2" "MCF7" "A549")
CHROM=("chr10-chr17" "chr11-chr7" "chr12-chr8" "chr13-chr9" "chr15-chr16")

for CELL in ${CELLLINE[@]}; do
        for CHRPAIR in ${CHROM[@]}; do
		echo -e $CELL"\t"$CHRPAIR
		# Cleanup in case of prior run
		rm -f parameter_$CELL\_$CHRPAIR\-1.slurm
                rm -f parameter_$CELL\_$CHRPAIR\-2.slurm
                rm -f parameter_$CELL\_$CHRPAIR\-3.slurm
		# Output header
		echo "$HEADER" > parameter_$CELL\_$CHRPAIR\-1.slurm
		echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"1\" --index=\"-1\" > $CELL\_$CHRPAIR\_1log.out &" >> parameter_$CELL\_$CHRPAIR\-1.slurm
		echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-1.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"1\" --index=\"-2\" > $CELL\_$CHRPAIR\_2log.out &" >> parameter_$CELL\_$CHRPAIR\-1.slurm
#                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-1.slurm
#                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"2\" --index=\"-1\" > $CELL\_$CHRPAIR\_3log.out &" >> parameter_$CELL\_$CHRPAIR\-1.slurm
#                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-1.slurm
#                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"2\" --index=\"-2\" > $CELL\_$CHRPAIR\_4log.out" >> parameter_$CELL\_$CHRPAIR\-1.slurm
		echo "wait" >> parameter_$CELL\_$CHRPAIR\-1.slurm

                echo "$HEADER" > parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"3\" --index=\"-1\" > $CELL\_$CHRPAIR\_5log.out &" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"3\" --index=\"-2\" > $CELL\_$CHRPAIR\_6log.out &" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"4\" --index=\"-1\" > $CELL\_$CHRPAIR\_7log.out &" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-2.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"4\" --index=\"-2\" > $CELL\_$CHRPAIR\_8log.out" >> parameter_$CELL\_$CHRPAIR\-2.slurm
		echo "wait" >> parameter_$CELL\_$CHRPAIR\-2.slurm

                echo "$HEADER" > parameter_$CELL\_$CHRPAIR\-3.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"5\" --index=\"-1\" > $CELL\_$CHRPAIR\_9log.out &" >> parameter_$CELL\_$CHRPAIR\-3.slurm
                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-3.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"5\" --index=\"-2\" > $CELL\_$CHRPAIR\_10log.out" >> parameter_$CELL\_$CHRPAIR\-3.slurm
                echo "wait" >> parameter_$CELL\_$CHRPAIR\-3.slurm

                echo "$HEADER" > parameter_$CELL\_$CHRPAIR\-4.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"6\" --index=\"-1\" > $CELL\_$CHRPAIR\_11log.out &" >> parameter_$CELL\_$CHRPAIR\-4.slurm
                echo "sleep 2" >> parameter_$CELL\_$CHRPAIR\-4.slurm
                echo "python $TRAIN --fileInput="\$tdir/data/CHR_NETWORK/" --parameter --cellLine=\"$CELL\" --chrPair=\"$CHRPAIR\" --model=\"6\" --index=\"-2\" > $CELL\_$CHRPAIR\_12log.out" >> parameter_$CELL\_$CHRPAIR\-4.slurm
                echo "wait" >> parameter_$CELL\_$CHRPAIR\-4.slurm


#		sbatch parameter_$CELL\_$CHRPAIR\-1.slurm
#		sbatch parameter_$CELL\_$CHRPAIR\-2.slurm
		sbatch parameter_$CELL\_$CHRPAIR\-3.slurm
#		sbatch parameter_$CELL\_$CHRPAIR\-4.slurm
	done

done
