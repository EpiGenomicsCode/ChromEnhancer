WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
MODELPATH=$WORKINGDIR/output-large/modelWeights
MODEL=LargeDataset_chr12-chr8_test_chr8_valid_chr12_model4_clkeep_K562_epoch_100.pt

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

SWARM=$WORKINGDIR/bin/swarm/swarm_analysis_large.py
MODELTYPE=4
MODELISZE=33000

LOGS=$WORKINGDIR/logs-large
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-large
mkdir -p $SLURM

# Cleanup in case of prior run
rm -f $SLURM/swarm_large*

for i in {1..10}; do

	OUTPUT=$WORKINGDIR/output-large/swarm/iter$i
	mkdir -p $OUTPUT

	slurmID=swarm_large-$i\.slurm
	echo "$HEADER" > $SLURM/$slurmID
	#time python $SWARM --modelPath $MODEL --modelSize $MODELISZE --modelType $MODELTYPE --outputPath $OUTPUT --randomSeed $i
	echo "time python $SWARM --modelPath \$tdir/$MODEL --modelSize $MODELISZE --modelType $MODELTYPE --outputPath $OUTPUT --randomSeed $i --epochs 50" >> $SLURM/$slurmID

done
exit
for file in $SLURM/swarm_large*; do
	sbatch $file
done
