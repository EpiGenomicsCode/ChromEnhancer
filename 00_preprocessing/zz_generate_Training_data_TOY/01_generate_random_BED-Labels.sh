module load bedtools

CELLLINE=("K562")
CHR1=("chr10")
CHR2=("chr17")

RANDBED=../../bin/toy_network/generate_random_BED-label.pl

for CELL in ${CELLLINE[@]}; do
	for (( i=0; i<${#CHR1[@]}; i++ )); do
		echo -e $CELL"\t"${CHR1[$i]}"\t"${CHR2[$i]}

                perl $RANDBED 100000 0.01 $CELL\_StringentEnhancer_${CHR1[$i]}\_TOY.bed
		cp $CELL\_StringentEnhancer_${CHR1[$i]}\_TOY.bed $CELL\_LenientEnhancer_${CHR1[$i]}\_TOY.bed
#                perl $RANDBED 100000 0.05 $CELL\_LenientEnhancer_${CHR1[$i]}\_TOY.bed
                perl $RANDBED 100000 0.01 $CELL\_StringentEnhancer_${CHR2[$i]}\_TOY.bed
		cp $CELL\_StringentEnhancer_${CHR2[$i]}\_TOY.bed $CELL\_LenientEnhancer_${CHR2[$i]}\_TOY.bed
#                perl $RANDBED 100000 0.05 $CELL\_LenientEnhancer_${CHR2[$i]}\_TOY.bed
                perl $RANDBED 750000 0.5 $CELL\_${CHR1[$i]}\-${CHR2[$i]}\_TOY_train.bed

		#awk -v seed=1234 'BEGIN{srand(seed)}{print rand(), $0}' final.bed | sort -n -k 1 | awk 'sub(/\S* /,"")' > $CELL\_${CHR1[$i]}\-${CHR2[$i]}\_train.bed
	done
done

# Organize files
mkdir -p ../../data/TOY-TRAIN
mv *train.bed ../../data/TOY-TRAIN/

mkdir -p ../../data/TOY-HOLDOUT
mv *Enhancer_*bed ../../data/TOY-HOLDOUT/
