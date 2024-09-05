#!/bin/bash
#SBATCH -A bbse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=24:00:00

set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

MODEL=model5

CHIP=chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1
IDIR=../output-cell/predictions
GTILE=$IDIR/CLD_study_-_test_-_valid_-_$MODEL\_clkeep_K562-HepG2-MCF7_$CHIP\_type-1_epoch_20.bed
RANDOMCOORD=../data/Random_Coord/hg38_25KRand_1000bp.bed.gz

EDIR=../figures/fig3/panelb/$MODEL/Intersect
PDIR=../tables/table1/ENCODE-Peaks

ODIR=../tables/table1/PeakEnrichment
[ -d $ODIR ] || mkdir -p $ODIR

# Iterate by cell line
CELL="HepG2"
# Outputs
BOUNDMETRICS=$ODIR/TF-BoundMetrics_$CELL\.txt

# Write header
echo $'ENCFF\tChIP-CellLine\tTF-Target\tEnh-CellLine\tTotal_Bin_Size\tEnh_Bin_Size\tRand_Bind_Size\tTotalPeaks\tPeakOverlap\tPeakOverlap_Rand' > $BOUNDMETRICS

# Count total genomic tiles
GENOMESIZE=`wc -l $GTILE | awk '{print $1}'`
# Count Random coordinate tiles
RANDSIZE=`gunzip -c $RANDOMCOORD | wc -l - | awk '{print $1}'`

# Handle dash in MCF7
CL=`echo $CELL | sed 's/-//g'`

ENHANCERFILE=$EDIR/$MODEL\_$CL.bed

# Count all tiles marked as enhancers
ENHANCERCOUNT=`wc -l $ENHANCERFILE | awk '{print $1}'`

for PEAKFILE in $PDIR/*.bed.gz;
do
	PEAK=`basename $PEAKFILE ".bed.gz"`
	
	ENCFF=`echo $PEAK | awk -F"_" '{print $1}'`
	CHIPCELL=`echo $PEAK | awk -F"_" '{print $3}'`
	TARGET=`echo $PEAK | awk -F"_" '{print $4}'`

	# Count all tiles with peak overlap
	PEAKCOUNT=`bedtools intersect -f 0.5 -wb -a $PEAKFILE -b $GTILE | wc -l | awk '{print $1}'`

	# Count enhancer tiles with peak overlap
	PEAKOVERLAP=`bedtools intersect -f 0.5 -wb -a $PEAKFILE -b $ENHANCERFILE | wc -l | awk '{print $1}'`

	# Count random tiles with peak overlap
	PEAKOVERLAP_RAND=`bedtools intersect -f 0.5 -wb -a $PEAKFILE -b $RANDOMCOORD | wc -l | awk '{print $1}'`

	# Write contingency values
	echo $ENCFF $CHIPCELL $TARGET $CL $GENOMESIZE $ENHANCERCOUNT $RANDSIZE $PEAKCOUNT $PEAKOVERLAP $PEAKOVERLAP_RAND | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10}' >> $BOUNDMETRICS

done	
