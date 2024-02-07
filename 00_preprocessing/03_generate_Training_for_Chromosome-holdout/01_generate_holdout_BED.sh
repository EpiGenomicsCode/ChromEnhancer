CELLLINE=("K562" "HepG2" "MCF7" "A549")
CHROM=("chr16" "chr17" "chr7" "chr8" "chr9" "chr10" "chr11" "chr12" "chr13" "chr14" "chr15" "chrX")

LABEL=../bin/label_BED_score.pl
SORT=../bin/sort_BED.pl
GENLABEL=../bin/generate_label_from_BED.pl

# Tiling BED file covering the entire genome
GENOMEBED=../../data/GRCh38_BED/GRCh38_1000bp.bed.gz

for CELL in ${CELLLINE[@]}; do
	# Cell type specific enhancers
	STRINGENTPEAKS=../../data/Enhancer_Coord/$CELL\_hg38_StringentEnhancer_1000bp.bed.gz
	LENIENTPEAKS=../../data/Enhancer_Coord/$CELL\_hg38_LenientEnhancer_1000bp.bed.gz

	for CHR in ${CHROM[@]}; do
		echo -e $CELL"\t"$CHR;
		# Remove holdout chromosome from enhancer and genome file
		zgrep "$CHR" $STRINGENTPEAKS > Speaks.bed
		zgrep "$CHR" $LENIENTPEAKS > Lpeaks.bed
		zgrep "$CHR" $GENOMEBED > genome.bed

		# Split chromosome into two files, enhancer and non-enhancer
		bedtools intersect -u -a genome.bed -b Speaks.bed > test-s.bed
		bedtools intersect -v -a genome.bed -b Speaks.bed > test-f.bed
		# Label split files as 0/1 depending on statust
		perl $LABEL test-s.bed 1 test-s_final
		perl $LABEL test-f.bed 0 test-f_final
		# Merge files back together
		cat test-s_final test-f_final > test.bed
		# Sort final file
		perl $SORT test.bed $CELL\_StringentEnhancer_$CHR\.bed

		# Split chromosome into two files, enhancer and non-enhancer
		bedtools intersect -u -a genome.bed -b Lpeaks.bed > test-s.bed
		bedtools intersect -v -a genome.bed -b Lpeaks.bed > test-f.bed
		# Label split files as 0/1 depending on statust
		perl $LABEL test-s.bed 1 test-s_final
		perl $LABEL test-f.bed 0 test-f_final
		# Merge files back together
		cat test-s_final test-f_final > test.bed
		# Sort final file
		perl $SORT test.bed $CELL\_LenientEnhancer_$CHR\.bed

		# Cleanup
		rm Speaks.bed Lpeaks.bed genome.bed test-s.bed test-f.bed test-s_final test-f_final test.bed
	done
done

# Generate label files
for file in *bed; do
	newFile="${file/.bed/.label}"
	perl $GENLABEL $file $newFile
done

# Organize files
mkdir -p ../../data/CHR-HOLDOUT
mv *.bed ../../data/CHR-HOLDOUT/
mv *label ../../data/CHR-HOLDOUT/
