#!/bin/bash

# Call enhancers by rank across 6 models, 2 replicates of each of 4 cell lines
#   1. Sort genomic tiles by enhancer scores
#   2. Call the top THRESH as enhancers
#   3. Perform both union and intersection to merge replicates

# |--CalledEnhancers
#   |--model1
#     |--model1_${CL}_${REP}_SORT-ByEnhancerScore.bed
#     |--model1_${CL}_${REP}-1per.bed
#     |--...
#     |--Intersect
#       |--model1_A549.bed
#       |--model1_HEPG2.bed
#       |--model1_K562.bed
#       |--model1_MCF7.bed
#     |--Union
#       |--DIFF_A549.bed
#       |--DIFF_HEPG2.bed
#       |--DIFF_K562.bed
#       |--DIFF_MCF7.bed
#       |--model1_A549.bed
#       |--model1_HEPG2.bed
#       |--model1_K562.bed
#       |--model1_MCF7.bed
#   |--model2
#     |--...
#   |--model3
#     |--...
#   |--model4
#     |--...
#   |--model5
#     |--...
#   |--model6
#     |--...


set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

CHIP=chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1
IDIR=../output-cell/predictions
OUTPUT=../figures/fig3/panelb

[ -d $OUTPUT ] || mkdir -p $OUTPUT

THRESH=1per

# Iterate model number
for MODEL in "model1" "model2" "model3" "model4" "model5" "model6" "model7";
do
	ODIR=$OUTPUT/$MODEL
	[ -d $ODIR ] || mkdir -p $ODIR
	[ -d $ODIR/Union ] || mkdir -p $ODIR/Union
	[ -d $ODIR/Intersect ] || mkdir -p $ODIR/Intersect

	# Iterate replicates
	for REP in "1" "2";
	do
		# Store CL-specific files as constants
		A549=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-MCF7_$CHIP\_type-$REP\_epoch_20.bed
		HEPG2=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-MCF7-A549_$CHIP\_type-$REP\_epoch_20.bed
		K562=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_HepG2-MCF7-A549_$CHIP\_type-$REP\_epoch_20.bed
		MCF7=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-A549_$CHIP\_type-$REP\_epoch_20.bed
		
		# Sort Enhancer BED files
		sort -grk5,5 $A549 > $ODIR/$MODEL\_A549_$REP\_SORT-ByEnhancerScore.bed
		sort -grk5,5 $HEPG2 > $ODIR/$MODEL\_HepG2_$REP\_SORT-ByEnhancerScore.bed
		sort -grk5,5 $K562 > $ODIR/$MODEL\_K562_$REP\_SORT-ByEnhancerScore.bed
		sort -grk5,5 $MCF7 > $ODIR/$MODEL\_MCF7_$REP\_SORT-ByEnhancerScore.bed

		# Call top 1 percent
		head -n 29596 $ODIR/$MODEL\_A549_$REP\_SORT-ByEnhancerScore.bed > $ODIR/$MODEL\_A549_$REP-$THRESH.bed
		head -n 29596 $ODIR/$MODEL\_HepG2_$REP\_SORT-ByEnhancerScore.bed > $ODIR/$MODEL\_HepG2_$REP-$THRESH.bed
		head -n 29596 $ODIR/$MODEL\_K562_$REP\_SORT-ByEnhancerScore.bed > $ODIR/$MODEL\_K562_$REP-$THRESH.bed
		head -n 29596 $ODIR/$MODEL\_MCF7_$REP\_SORT-ByEnhancerScore.bed > $ODIR/$MODEL\_MCF7_$REP-$THRESH.bed

	done

	# Iterate cell lines
	for CL in "A549" "HepG2" "K562" "MCF7";
	do
		EFILE=$ODIR/$MODEL\_$CL

		# Build intersect set (use bedtools intersect)
		bedtools intersect -f 1.0 -wa -a $EFILE\_1-$THRESH.bed -b $EFILE\_2-$THRESH.bed > $ODIR/Intersect/$MODEL\_$CL.bed

		# Build union set (use bedtools intersect)
		bedtools intersect -f 1.0 -wa -v -a $EFILE\_1-$THRESH.bed -b $EFILE\_2-$THRESH.bed > $ODIR/Union/DIFF_$CL.bed
		cat $ODIR/Intersect/$MODEL\_$CL.bed $ODIR/Union/DIFF_$CL.bed | sort -grk5,5 > $ODIR/Union/$MODEL\_$CL.bed

		done

done
