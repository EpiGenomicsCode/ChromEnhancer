CELLLINE=("K562" "HepG2" "MCF7" "A549")

LABEL=job/label_BED_score.pl
SORT=job/sort_BED.pl
GENLABEL=job/generate_label_from_BED.pl
SPLIT=job/split_BED.pl

# Tiling BED file covering the entire genome
GENOMEBED=GRCh38_BED/GRCh38_1000bp.bed.gz

for CELL in ${CELLLINE[@]}; do
         # Cell type specific enhancers
			 	 STRINGENTPEAKS=../02_call_Enhancers/Enhancer_Coord/$CELL\_hg38_StringentEnhancer_1000bp.bed
			   LENIENTPEAKS=../02_call_Enhancers/Enhancer_Coord/$CELL\_hg38_LenientEnhancer_1000bp.bed
         echo -e $CELL;
         # Split chromosome into two files, enhancer and non-enhancer
         bedtools intersect -u -a $GENOMEBED -b $STRINGENTPEAKS > test-s.bed
         bedtools intersect -v -a $GENOMEBED -b $STRINGENTPEAKS > test-f.bed
         # Label split files as 0/1 depending on statust
         perl $LABEL test-s.bed 1 test-s_final
         perl $LABEL test-f.bed 0 test-f_final
         # Merge files back together
         cat test-s_final test-f_final > test.bed
         # Sort final file
         perl $SORT test.bed $CELL\_StringentEnhancer.bed
         perl $SPLIT $CELL\_StringentEnhancer.bed $CELL\_enhancer

         # Split chromosome into two files, enhancer and non-enhancer
         bedtools intersect -u -a $GENOMEBED -b $LENIENTPEAKS > test-s.bed
         bedtools intersect -v -a $GENOMEBED -b $LENIENTPEAKS > test-f.bed
         # Label split files as 0/1 depending on statust
         perl $LABEL test-s.bed 1 test-s_final
         perl $LABEL test-f.bed 0 test-f_final
         # Merge files back together
         cat test-s_final test-f_final > test.bed
         # Sort final file
         perl $SORT test.bed $CELL\_LenientEnhancer.bed

         # Cleanup
         rm test-s.bed test-f.bed test-s_final test-f_final test.bed
done

# Generate label files
for file in *Enhancer.bed; do
	newFile="${file/.bed/.label}"
	perl $GENLABEL $file $newFile
done

# Organize files
gzip *.bed
mkdir -p HOLDOUT
mv *.bed.gz HOLDOUT/
mv *label HOLDOUT/
