#!/bin/bash

# Calculate ENCODE ChIP peak enrichment at enhancer calls from model 6
#   1. Intersect ENCODE peak files with enhancer tiles to get contingency table counts of genomic tile overlaps
#   2. Perform chi-square test for independence (between peak calls and enhancer calls) with Benjamini-Hochberg multiple test correction for each row of contingency table values
#   3. Reshape the corrected p-value statistics into a table with TF target per row and cell line per column for heatmap visualization
#   4. Perform hierarchical clustering on independence p-values

# |--ContingencyTableMetrics.txt
# |--ChiSquareMetrics.txt
# |--ChiSquareMetrics_RESHAPE.txt
# |--ChiSquareMetrics_RESHAPE_CLUSTER.txt
# |--ChiSquareMetrics_RESHAPE_CLUSTER.png


set -exo
module load bedtools

MODEL=model6

CHIP=chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1
IDIR=/storage/group/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen/output-cell/predictions
IDIR=../output-cell/predictions
GTILE=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-MCF7_$CHIP\_type-1_epoch_20.bed

EDIR=CalledEnhancers/$MODEL/Intersect
PDIR=ENCODE-Peaks

ODIR=PeakEnrichment
[ -d $ODIR ] || mkdir $ODIR

# Outputs
CONTINGENCY=$ODIR/ContingencyTableMetrics.txt

# Script shortcuts
SCRIPTMANAGER=../bin/ScriptManager-v0.14-dev.jar
CHISQUARE=bin/chisq_test_from_contingency_table.py
RESHAPE=bin/pivot_table_by_index.py 
HIERARCHICAL=bin/hierarchical-2d.py

# Write header
echo $'ENCFF\tCellLine\tTF-Target\tNoPeak-NoEnhancer\tPeak-NoEnhancer\tNoPeak-Enhancer\tPeak-Enhancer' > $CONTINGENCY

# Count total genomic tiles
T_XX=`wc -l $GTILE | awk '{print $1}'`

# Iterate by cell line
for CELL in "A549" "HepG2" "K562" "MCF-7";
do
	# Handle dash in MCF7
	CL=`echo $CELL | sed 's/-//g'`
	
	ENHANCERFILE=$EDIR/$MODEL\_$CL.bed
	
	# Count all tiles marked as enhancers
	T_XE=`wc -l $ENHANCERFILE | awk '{print $1}'`

	for PEAKFILE in $PDIR/*$CELL*.bed.gz;
	do
		PEAK=`basename $PEAKFILE ".bed.gz"`
		
		ENCFF=`echo $PEAK | awk -F"_" '{print $1}'`
		TARGET=`echo $PEAK | awk -F"_" '{print $4}'`
	
		# Count all tiles with peak overlap
		T_PX=`bedtools intersect -f 0.5 -wb -a $PEAKFILE -b $GTILE | wc -l | awk '{print $1}'`

		# Count enhancer tiles with peak overlap
		N_PE=`bedtools intersect -f 0.5 -wb -a $PEAKFILE -b $ENHANCERFILE | wc -l | awk '{print $1}'`

		# Calculate contingency values
		N_NE=$(($T_XE-$N_PE))
		N_PN=$(($T_PX-$N_PE))
		N_NN=$(($T_XX-$N_NE-$N_PE-$N_PN))

		# Write contingency values
		echo $ENCFF $CL $TARGET $N_NN $N_PN $N_NE $N_PE | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,$7,$8}' >> $CONTINGENCY

	done	
done

# Check independence (per row basis)
python $CHISQUARE -i $CONTINGENCY -o $ODIR/ChiSquareMetrics.txt
# creates table: CL - Target - ChiSqStat - p-value - adjusted p-value - enriched?

# Filter by p-value?

# Reorganize by CL - reshape table by CL into p-value matrix and fill blanks with NaNs
python $RESHAPE -i $ODIR/ChiSquareMetrics.txt -o $ODIR/ChiSquareMetrics_PIVOT.txt -c 2 -x 1 -v 6

# Heatmap
java -jar $SCRIPTMANAGER figure-generation three-color -an 0 -ax 1 -x 500 -y 100 -l 1 -r 1 \
	$ODIR/ChiSquareMetrics_PIVOT.txt \
	-o $ODIR/ChiSquareMetrics_PIVOT.png

# Hierarchical cluster 
python $HIERARCHICAL -i $ODIR/ChiSquareMetrics_PIVOT.txt -o $ODIR/ChiSquareMetrics_PIVOT_CLUSTER.txt

# Heatmap
java -jar $SCRIPTMANAGER figure-generation three-color -an 0 -ax 1 -x 500 -y 100 -l 1 -r 1 \
	$ODIR/ChiSquareMetrics_PIVOT_CLUSTER.txt \
	-o $ODIR/ChiSquareMetrics_PIVOT_CLUSTER.png

