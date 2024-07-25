WORKINGDIR=/scratch/bbse/wklai/EnhancerNN/ChromEnhancer
DATADIR=$WORKINGDIR/data/CELL_NETWORK
HEADER="#!/bin/bash
#SBATCH -A bbse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --time=48:00:00

module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/captum
cd $WORKINGDIR
tdir="$(mktemp -d /tmp/foo.XXXXXXXXX)"
mkdir -p \$tdir
cp -r $DATADIR \$tdir/
"

SHAP=$WORKINGDIR/bin/xai/calculate_XAI.py

MODELPATH=$WORKINGDIR/output-cell/modelWeights

LOGS=$WORKINGDIR/logs-cell
mkdir -p $LOGS
SLURM=$WORKINGDIR/slurm-cell
mkdir -p $SLURM
OUTPUT=$WORKINGDIR/output-cell/xai
mkdir -p $OUTPUT

# Cleanup in case of prior run
rm -f $SLURM/xai_CLD-*

COUNT=0
regex='model([0-9]+)_clkeep_([A-Za-z0-9-]+)_chkeep_.*_type-([0-9]+)_epoch_[0-9]+\.pt'

for modelFile in $MODELPATH/*K562-HepG2-A549*type-2*pt; do
        if [[ $modelFile =~ $regex ]]; then
                model_number="${BASH_REMATCH[1]}"
                type_number="${BASH_REMATCH[3]}"

		slurmID=xai_CLD-$COUNT\.slurm
		echo "$HEADER" > $SLURM/$slurmID
		echo "python $SHAP --fileInput="\$tdir/CELL_NETWORK/" --modelPath $modelFile --modelType $model_number --dataType -$type_number --outputPath=$OUTPUT --cellLine K562 HepG2 A549" >> $SLURM/$slurmID

		((COUNT++))
	fi

done
exit

cd $SLURM
for file in xai_CLD*slurm; do
        sbatch $file;
done
