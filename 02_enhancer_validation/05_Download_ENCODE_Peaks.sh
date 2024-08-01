#!/bin/bash

# Download ENCODE ChIP peak files for processing in the next step

# |--ENCODE-Peaks
#   |--{ENCFF}_ChIP-seq_{CELLINE}_{TF}_peaks.bed.gz
#   |--...


set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

METADATA=ENCODE_Peak_Metadata.txt
ODIR=../figures/fig3/panelc/ENCODE-Peaks
[ -d $ODIR ] || mkdir -p $ODIR

for SLURM_ARRAY_TASK_ID in {1..1449};
do
	INDEX=$((SLURM_ARRAY_TASK_ID+1))

	# Parse metadata
	INFO=`sed "${INDEX}q;d" $METADATA`
	ENCFF=`echo $INFO | awk '{print $1}'`
	ASSAY=`echo $INFO | awk '{print $5}'`
	CELL=`echo $INFO | awk '{print $3}'`
	TARGET=`echo $INFO | awk '{print $4}'`
	GBUILD=`echo $INFO | awk '{print $6}'`

	# Construct output filename from metadata
	OUTPUT=$ODIR/$ENCFF\_$ASSAY\_$CELL\_$TARGET\_peaks.bed.gz

	# Construct ENCODE data download URL
	HREF=/files/$ENCFF/@@download/$ENCFF.bed.gz

	# Download data
	echo "Fetching from https://www.encodeproject.org$HREF"
	wget -c -O $OUTPUT https://www.encodeproject.org$HREF

done
