WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer/data/CELL_NETWORK
HEADER="#!/bin/bash
#SBATCH -A bbse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=2:00:00

module load anaconda3_cpu
cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir -p \$tdir
cp -r $DATADIR \$tdir/
"

PREDICT=$WORKINGDIR/calculate_EnhancerProb.py
UPDATE=$WORKINGDIR/bin/update_BED_with_EnhPred_scores.py

#python calculate_EnhancerProb.py --fileInput /scratch/bbse/wklai/EnhancerNN/ChromEnhancer/data/CELL_NETWORK/ --modelPath output-cell/modelWeights/CLD_study_-_test_-_valid_-_model1_clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-1_epoch_20.pt --modelType 1 --fileOutput test.tab --cellLine K562 HepG2 A549

MODELPATH=$WORKINGDIR/output-cell/modelWeights
BEDPATH=$WORKINGDIR/data/CELL-HOLDOUT

LOGS=$WORKINGDIR/logs-cell
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-cell
mkdir -p $SLURM
OUTPUT=$WORKINGDIR/output-cell/predictions
mkdir -p $OUTPUT

#regex='model([0-9]+)_clkeep_([A-Za-z0-9-]+)'
regex='model([0-9]+)_clkeep_([A-Za-z0-9-]+)_chkeep_.*_type-([0-9]+)_epoch_[0-9]+\.pt'
COUNT=0

# Cleanup in case of prior run
rm -f $SLURM/pred_CLD-*

for modelFile in $MODELPATH/*pt; do
	if [[ $modelFile =~ $regex ]]; then
		model_number="${BASH_REMATCH[1]}"
		cell_lines="${BASH_REMATCH[2]}"
		type_number="${BASH_REMATCH[3]}"
	        cell_lines_with_spaces=$(echo "$cell_lines" | sed 's/-/ /g')

		# Get the basename of the file (with extension)
		basename_with_ext=$(basename "$modelFile")
		# Remove the extension
		basename_without_ext="${basename_with_ext%.*}"

		slurmID=pred_CLD-$COUNT\.slurm
		echo "$HEADER" > $SLURM/$slurmID
		echo "python $PREDICT --fileInput="\$tdir/CELL_NETWORK/" --modelPath $modelFile --modelType $model_number --dataType -$type_number --fileOutput=$OUTPUT/$basename_without_ext\_PRED.tab --cellLine $cell_lines_with_spaces > $LOGS/CLD-$COUNT\_log_model$Mmodel_number\.out" >> $SLURM/$slurmID
		echo "CELL=\$(head -n 1 $OUTPUT/$basename_without_ext\_PRED.tab | awk -F\"'\" '{print \$2}')"  >> $SLURM/$slurmID
		echo "python $UPDATE $BEDPATH/*\$CELL*bed.gz $OUTPUT/$basename_without_ext\_PRED.tab $OUTPUT/$basename_without_ext\.bed" >> $SLURM/$slurmID

	        ((COUNT++))
	fi
done

cd $SLURM
for file in pred_CLD*slurm; do
	sbatch $file;
done
