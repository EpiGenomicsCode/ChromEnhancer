# Reference data
GRCH38=input/GRCh38.chrom.sizes
BLACKLIST=input/ENCFF356LFX.bed.gz
CHRY=input/hg38_chrY.bed

# Script to tile the genome
TILE=../../bin/tile_Genome.pl
SORT=../../bin/sort_BED.pl

# Set size of tiles
SIZE=1000

# Split the genome into tiled chromosome files
perl $TILE $GRCH38 1000 GRCh38
cat *.bed > GRCh38_1000bp.bed

# Remove blacklist regions from genome to prevent their inclusion in training
bedtools intersect -v -a GRCh38_1000bp.bed -b $BLACKLIST > temp.bed

# Remove chrY to standardize chromosome content
bedtools intersect -v -a temp.bed -b $CHRY > GRCh38_1000bp_filter.bed

perl $SORT GRCh38_1000bp_filter.bed GRCh38_1000bp_sort.bed
mv GRCh38_1000bp_sort.bed GRCh38_1000bp.bed

# Compress final file
gzip GRCh38_1000bp.bed

# Organize data
mkdir -p ../../data/GRCh38_BED
mv *.gz ../../data/GRCh38_BED/

# Cleanup
rm *.bed
