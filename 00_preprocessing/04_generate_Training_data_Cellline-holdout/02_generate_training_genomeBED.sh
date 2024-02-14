module load bedtools

CELLLINE=("K562" "HepG2" "MCF7" "A549")

TILESUCCESS=../bin/tile_Success_BED.pl
TILEFAILURE=../bin/flank_Failure_BED.pl
LABEL=../bin/label_BED_score.pl
SORT=../bin/sort_BED.pl
GENLABEL=../bin/generate_label_from_BED.pl
SPLIT=../bin/split_BED.pl

# Tiling BED file covering the entire genome
GENOMEBED=../../data/GRCh38_BED/GRCh38_1000bp.bed.gz

COUNTER=0
for CELL in ${CELLLINE[@]}; do
        # Cell type specific enhancers
	TESTPEAKS=../../data/Enhancer_Coord/$CELL\_hg38_StringentEnhancer_1000bp.bed.gz

	echo -e $CELL
	# Get overlap of enhancers with genomic tiles
	bedtools intersect -u -a $GENOMEBED -b $TESTPEAKS > peak_train.bed
        # For each original enhancer coordinate, generate 10 offset windows, 200bp total, 20 bp frameshift
	perl $TILESUCCESS peak_train.bed train_peaks-s.bed
	# For each original enhancer coordinate, generate flanking coordinates 5000bp away on each side
	bedtools merge -i peak_train.bed > peak_train-merge.bed
	perl $TILEFAILURE peak_train-merge.bed 5000 1000 train_peaks-f.bed
	cat train_peaks-s.bed train_peaks-f.bed > train_peaks-all.bed
	#Generate random peaks from genome that are not peaks
	bedtools intersect -v -a $GENOMEBED -b train_peaks-all.bed > remaining_peaks.bed
	ROWS=$(wc -l train_peaks-s.bed | awk '{print $1}')
        shuf --random-source=$GENOMEBED -n $ROWS remaining_peaks.bed > train_peaks-neg_raw.bed
	perl $LABEL train_peaks-neg_raw.bed 0 train_peaks-neg.bed
	cat train_peaks-all.bed train_peaks-neg.bed > all_train.bed
	# Remove any training regions that DO NOT exist in the reference genome
	bedtools intersect -u -a all_train.bed -b $GENOMEBED > final.bed
	# Randomize file
	awk -v seed=1234 'BEGIN{srand(seed)}{print rand(), $0}' final.bed | sort -n -k 1 | awk 'sub(/\S* /,"")' > $CELL\_train.bed
        perl $SPLIT $CELL\_train.bed $CELL\_train

	# Increment counter/random seed
	let COUNTER++

done

# Remove temporary files
rm peak_train.bed train_peaks-s.bed peak_train-merge.bed train_peaks-f.bed train_peaks-all.bed remaining_peaks.bed train_peaks-neg_raw.bed train_peaks-neg.bed all_train.bed final.bed

# Generate label files
for file in *bed; do
        newFile="${file/.bed/.label}"
        perl $GENLABEL $file $newFile
done

# Organize files
gzip *.bed
mkdir -p ../../data/CELL-TRAIN/
mv *.bed.gz ../../data/CELL-TRAIN/
mv *label ../../data/CELL-TRAIN/
