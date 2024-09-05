WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
MODELPATH=$WORKINGDIR/output-large/modelWeights
MODEL=LargeDataset1_6_epoch_100.pt

HEADER="#!/bin/bash
#SBATCH -A bbse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=8:00:00

module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/captum

cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir \$tdir
cp -r $MODELPATH/$MODEL \$tdir/
"

SWARM=$WORKINGDIR/bin/swarm/swarm_analysis_large_inverse.py
MODELTYPE=6
MODELISZE=33000

LOGS=$WORKINGDIR/logs-large
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-large
mkdir -p $SLURM

# Cleanup in case of prior run
rm -f $SLURM/swarm_inverse-large*

for i in {1..10}; do

	OUTPUT=$WORKINGDIR/output-large/swarm_inverse/iter$i
	mkdir -p $OUTPUT

	slurmID=swarm_invsere-large-$i\.slurm
	echo "$HEADER" > $SLURM/$slurmID
	#time python $SWARM --modelPath $MODEL --modelSize $MODELISZE --modelType $MODELTYPE --outputPath $OUTPUT --randomSeed $i
	echo "time python $SWARM --modelPath \$tdir/$MODEL --modelSize $MODELISZE --modelType $MODELTYPE --outputPath $OUTPUT --randomSeed $i --epochs 50" >> $SLURM/$slurmID

done
exit
for file in $SLURM/swarm_inverse*; do
	sbatch $file
done
