#!/bin/bash

# Call enhancers by rank across 6 models, 2 replicates of each of 4 cell lines
#   1. Perform intersection between cell line-specific enhancer calls and update group id to reflect overlaps
#   2. In final merge, set genomic tile RefPT to midpoint of the 1 kb tiles
#   3. Split genomic tiles by group id
#   3. Create Venn diagram figure from the group counts

# |--VennLabel
#   |--model1
#     |--Diagram
#       |--Intersections_4.txt
#       |--Venn_4.svg
#     |--A549_K562.bed
#     |--HepG2_MCF7.bed
#     |--Venn-1per.txt
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

SPLIT=bin/split_file_by_category.py
VENN=../bin/chart/venn4py_CellLineParse.py

OUTPUT=../figures/fig3/panelb
[ -d $OUTPUT ] || mkdir -p $OUTPUT

# Iterate model number
for MODEL in "model1" "model2" "model3" "model4" "model5" "model6" "model7";
do
	IDIR=$OUTPUT/$MODEL

	VDIR=$OUTPUT/$MODEL
	[ -d $VDIR ] || mkdir -p $VDIR
	
	# Do cross-cell line intersections
	A549=$IDIR/Intersect/$MODEL\_A549.bed
	HEPG2=$IDIR/Intersect/$MODEL\_HepG2.bed
	K562=$IDIR/Intersect/$MODEL\_K562.bed
	MCF7=$IDIR/Intersect/$MODEL\_MCF7.bed

	# A549 x K562
	bedtools intersect -f 1.0 -wa    -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549;K562",$5,$6}' > $VDIR/A549_K562.bed
	bedtools intersect -f 1.0 -wa -v -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549",     $5,$6}' >> $VDIR/A549_K562.bed
	bedtools intersect -f 1.0 -wa -v -a $K562  -b $A549  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"K562",     $5,$6}' >> $VDIR/A549_K562.bed

	# MCF7 x HepG2
	bedtools intersect -f 1.0 -wa    -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2;MCF7",$5,$6}' > $VDIR/HepG2_MCF7.bed
	bedtools intersect -f 1.0 -wa -v -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2",     $5,$6}' >> $VDIR/HepG2_MCF7.bed
	bedtools intersect -f 1.0 -wa -v -a $MCF7  -b $HEPG2 | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"MCF7",      $5,$6}' >> $VDIR/HepG2_MCF7.bed

	# Merge
	bedtools intersect -f 1.0 -wo    -a $VDIR/A549_K562.bed  -b $VDIR/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4";"$10}' > $VDIR/Venn-1per.txt
	bedtools intersect -f 1.0 -wa -v -a $VDIR/A549_K562.bed  -b $VDIR/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $VDIR/Venn-1per.txt
	bedtools intersect -f 1.0 -wa -v -a $VDIR/HepG2_MCF7.bed -b $VDIR/A549_K562.bed  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $VDIR/Venn-1per.txt

	# Venn split
	#python $SPLIT $VDIR/Venn-1per.txt $VDIR/VennSplit 1
	
	# Make venn figure
	[ -d $VDIR/Diagram ] || mkdir $VDIR/Diagram
	python $VENN -i $VDIR/Venn-1per.txt -o $VDIR/Diagram

done
