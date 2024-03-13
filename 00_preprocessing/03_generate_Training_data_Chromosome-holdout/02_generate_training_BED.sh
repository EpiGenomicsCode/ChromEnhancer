module load bedtools

CELLLINE=("K562" "HepG2" "MCF7" "A549")
CHR1=("chr10" "chr11" "chr12" "chr13" "chr14" "chr15")
CHR2=("chr17" "chr7" "chr8" "chr9" "chrX" "chr16")

TILESUCCESS=../../bin/tile_Success_BED.pl
TILEFAILURE=../../bin/flank_Failure_BED.pl
LABEL=../../bin/label_BED_score.pl
SORT=../../bin/sort_BED.pl

# Tiling BED file covering the entire genome
GENOMEBED=../../data/GRCh38_BED/GRCh38_1000bp.bed.gz

for CELL in ${CELLLINE[@]}; do
	# Cell type specific enhancers
	TESTPEAKS=../../data/Enhancer_Coord/$CELL\_hg38_StringentEnhancer_1000bp.bed.gz

	for (( i=0; i<${#CHR1[@]}; i++ )); do
		echo -e $CELL"\t"${CHR1[$i]}"\t"${CHR2[$i]}
		# Remove holdout chromosomes
		zgrep -v -e "${CHR1[$i]}" -e "${CHR2[$i]}" $GENOMEBED > genome_train.bed
		zgrep -v -e "${CHR1[$i]}" -e "${CHR2[$i]}" $TESTPEAKS > peaks.bed
		# Get overlap of enhancers with genomic tiles
		bedtools intersect -u -a $GENOMEBED -b peaks.bed > peak_train.bed
		# For each original enhancer coordinate, generate 10 offset windows, 200bp total, 20 bp frameshift
		perl $TILESUCCESS peak_train.bed train_peaks-s.bed
		# For each original enhancer coordinate, generate flanking coordinates 5000bp away on each side
		bedtools merge -i train_peaks-s.bed > peak_train-merge.bed
		perl $TILEFAILURE peak_train-merge.bed 5000 1000 train_peaks-f.bed
		cat train_peaks-s.bed train_peaks-f.bed > train_peaks-all.bed
		#Generate random peaks from genome that are not peaks
		bedtools intersect -v -a genome_train.bed -b train_peaks-all.bed > remaining_peaks.bed
		ROWS=$(wc -l train_peaks-s.bed | awk '{print $1}')
		shuf --random-source=$GENOMEBED -n $ROWS remaining_peaks.bed > train_peaks-neg_raw.bed
		perl $LABEL train_peaks-neg_raw.bed 0 train_peaks-neg.bed
		cat train_peaks-all.bed train_peaks-neg.bed > all_train.bed
		# Remove any training regions that DO NOT exist in the reference genome
		bedtools intersect -u -a all_train.bed -b $GENOMEBED > final.bed
		# Randomize file
		awk -v seed=1234 'BEGIN{srand(seed)}{print rand(), $0}' final.bed | sort -n -k 1 | awk 'sub(/\S* /,"")' > $CELL\_${CHR1[$i]}\-${CHR2[$i]}\_train.bed
	done
done

# Remove temporary files
rm genome_train.bed peaks.bed peak_train.bed train_peaks-s.bed peak_train-merge.bed train_peaks-f.bed train_peaks-all.bed remaining_peaks.bed train_peaks-neg_raw.bed train_peaks-neg.bed all_train.bed final.bed

# Organize files
mkdir -p ../../data/CHR-TRAIN
mv *bed ../../data/CHR-TRAIN/
