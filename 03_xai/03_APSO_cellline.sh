set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/captum

WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=$WORKINGDIR/data/CELL_NETWORK
MODELPATH=$WORKINGDIR/output-cell/modelWeights
MODEL=$MODELPATH/CLD_study_-_test_-_valid_-_model5_clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-2_epoch_20.pt

SWARM=../swarm_analysis.py
MODELTYPE=5
MODELISZE=800

for i in {1..10}; do

	OUTPUT=$WORKINGDIR/output-cell/swarm/iter$i
	mkdir -p $OUTPUT
	time python $SWARM --modelPath $MODEL --modelSize $MODELISZE --modelType $MODELTYPE --outputPath $OUTPUT --randomSeed $i
	
done
