#CTCF, H3K4me3, H3K27ac, p300, PolII, H3K36me3, H3K27me3, H3K4me1

HOLDOUT=../../data/TOY-HOLDOUT
TRAIN=../../data/TOY-TRAIN

CELLLINE=("K562")
CHR1=("chr10")
CHR2=("chr17")

POS=../../bin/toy_network/generate_rand-Chromtrack_POS.pl
NEG=../../bin/toy_network/generate_rand-Chromtrack_NEG.pl
UNI=../../bin/toy_network/generate_rand-Chromtrack.pl 

for file in $HOLDOUT/*StringentEnhancer*.bed; do
	echo $file

	CELL=$(echo $file | awk -F'/' '{print $5}' | cut -d'_' -f1)
	ENH=$(echo $file | awk -F'/' '{print $5}' | cut -d'_' -f2)
	CHROM=$(echo $file | awk -F'/' '{print $5}' | cut -d'_' -f3)

	perl $POS $file $HOLDOUT/$CELL\_$CHROM\_p300-1_combined.chromtrack
	perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_CTCF-1_combined.chromtrack
	perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_H3K27ac-1_combined.chromtrack
	perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_H3K4me3-1_combined.chromtrack
	perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_PolII-1_combined.chromtrack
        perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_H3K36me3-1_combined.chromtrack
        perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_H3K27me3-1_combined.chromtrack
        perl $NEG $file $HOLDOUT/$CELL\_$CHROM\_H3K4me1-1_combined.chromtrack

	perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_p300-2_combined.chromtrack
	perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_CTCF-2_combined.chromtrack
	perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_H3K27ac-2_combined.chromtrack
	perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_H3K4me3-2_combined.chromtrack
	perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_PolII-2_combined.chromtrack
        perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_H3K36me3-2_combined.chromtrack
        perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_H3K27me3-2_combined.chromtrack
        perl $UNI $file $HOLDOUT/$CELL\_$CHROM\_H3K4me1-2_combined.chromtrack
done

for file in $TRAIN/*.bed; do
        echo $file
	#../../data/TOY-TRAIN/K562_chr10-chr17_TOY_train.bed

        CELL=$(echo $file | awk -F'/' '{print $5}' | cut -d'_' -f1)
        CHROM=$(echo $file | awk -F'/' '{print $5}' | cut -d'_' -f2)

        perl $POS $file $TRAIN/$CELL\_$CHROM\_train_p300-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_CTCF-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_H3K27ac-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_H3K4me3-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_PolII-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_H3K36me3-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_H3K27me3-1_combined.chromtrack
        perl $NEG $file $TRAIN/$CELL\_$CHROM\_train_H3K4me1-1_combined.chromtrack

	perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_p300-2_combined.chromtrack
	perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_CTCF-2_combined.chromtrack
	perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_H3K27ac-2_combined.chromtrack
	perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_H3K4me3-2_combined.chromtrack
	perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_PolII-2_combined.chromtrack
        perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_H3K36me3-2_combined.chromtrack
        perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_H3K27me3-2_combined.chromtrack
        perl $UNI $file $TRAIN/$CELL\_$CHROM\_train_H3K4me1-2_combined.chromtrack

done

gzip -f $HOLDOUT/*chromtrack
gzip -f $TRAIN/*chromtrack
