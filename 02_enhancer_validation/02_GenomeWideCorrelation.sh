#!/bin/bash

# Generate Figure 3A - Genome wide correlation matrix of enhancer scores across 6 models, 2 replicates of each of 4 cell lines
#   1. Paste together genome wide scores into a matrix file (48 columns, genome tile rows)
#   2. Calculate correlation matrix
#   3. Perform 2D hierarchical clustering
#   4. Repeat on a per-model basis

# |--GenomeWideCorrelation
#   |--model1_aggregate.tab
#   |--model2_aggregate.tab
#   |--model3_aggregate.tab
#   |--model4_aggregate.tab
#   |--model5_aggregate.tab
#   |--model6_aggregate.tab
#   |--QualityControl-ID-Matching.txt
#   |--tile.txt
#   |--tile-header.txt
#   |--Total-48_EnhancerScores.tab
#   |--Total-48_EnhancerScores_CORRELATE.tab
#   |--Total-48_EnhancerScores_CORRELATE.svg
#   |--Total-48_EnhancerScores_CORRELATE_label.tab

#set -exo
module load anaconda3_cpu

# Inputs
CHIP=chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1
IDIR=../output-cell/predictions

ODIR=../figures/fig3/panela
[ -d $ODIR ] || mkdir -p $ODIR

# Outputs
TILE=$ODIR/tile.txt
HTILE=$ODIR/tile-header.txt
touch $HTILE

QCFILE=$ODIR/QualityControl-ID-Matching.txt
[[ -f $QCFILE ]] && rm $QCFILE

# Script shortcuts
CORRELATION=bin/correlation-matrix.py
HIERARCHICAL=bin/plot_hierarchical-2d.py

# Iterate model number
for MODEL in "model1" "model2" "model3" "model4" "model5" "model6" "model7";
do
	TEMP=/tmp/enhscore
	[ -d $TEMP ] || mkdir -p $TEMP

	# Iterate replicates
	for REP in "1" "2";
	do
		# Store CL-specific files as constants
		A549=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-MCF7_$CHIP\_type-$REP\_epoch_20.bed
		K562=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_HepG2-MCF7-A549_$CHIP\_type-$REP\_epoch_20.bed
		HEPG2=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-MCF7-A549_$CHIP\_type-$REP\_epoch_20.bed
		MCF7=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-A549_$CHIP\_type-$REP\_epoch_20.bed
	
		# Pull Enhancer scores
		cat <(echo "${MODEL}_A549_$REP") <(cut -f5 $A549) > $TEMP/A549_$REP.txt
		cat <(echo "${MODEL}_K562_$REP") <(cut -f5 $K562) > $TEMP/K562_$REP.txt
		cat <(echo "${MODEL}_HepG2_$REP") <(cut -f5 $HEPG2) > $TEMP/HepG2_$REP.txt
		cat <(echo "${MODEL}_MCF7_$REP") <(cut -f5 $MCF7) > $TEMP/MCF7_$REP.txt

		# Pull ID values
		paste $TILE <(cut -f4 $A549) | awk '{if ($1!=$2) print}' > $TEMP/$MODEL\_A549_$REP\.id
		paste $TILE <(cut -f4 $K562) | awk '{if ($1!=$2) print}' > $TEMP/$MODEL\_K562_$REP\.id
		paste $TILE <(cut -f4 $HEPG2) | awk '{if ($1!=$2) print}' > $TEMP/$MODEL\_HepG2_$REP\.id
		paste $TILE <(cut -f4 $MCF7) | awk '{if ($1!=$2) print}' > $TEMP/$MODEL\_MCF7_$REP\.id

	done
	# Concatenate model-specific files
	paste $HTILE \
		$TEMP/A549_1.txt $TEMP/A549_2.txt \
		$TEMP/K562_1.txt $TEMP/K562_2.txt \
		$TEMP/HepG2_1.txt $TEMP/HepG2_2.txt \
		$TEMP/MCF7_1.txt $TEMP/MCF7_2.txt \
		> $ODIR/$MODEL\_aggregate.tab

	# Write Quality control metrics	from ID values (sanity check, should indicate 0 for all resulting data)
	wc -l $TEMP/$MODEL\_*.id >> $QCFILE

	rm -r $TEMP
done

# Aggregate model-specific files
TOTAL=$ODIR/Total-48_EnhancerScores
paste $ODIR/tile-header.txt \
	<(cut -f2-9 $ODIR/model1_aggregate.tab) \
	<(cut -f2-9 $ODIR/model2_aggregate.tab) \
	<(cut -f2-9 $ODIR/model3_aggregate.tab) \
	<(cut -f2-9 $ODIR/model4_aggregate.tab) \
	<(cut -f2-9 $ODIR/model5_aggregate.tab) \
	<(cut -f2-9 $ODIR/model6_aggregate.tab) \
        <(cut -f2-9 $ODIR/model7_aggregate.tab) \
	> $TOTAL.tab

# Correlate and cluster
python $CORRELATION -i $TOTAL.tab -o $TOTAL\_CORRELATE.tab
python $HIERARCHICAL -i $TOTAL\_CORRELATE.tab -o $TOTAL\_CORRELATE.svg -t $TOTAL\_CORRELATE_label.tab

# Clean-up
rm $ODIR/model*.tab
