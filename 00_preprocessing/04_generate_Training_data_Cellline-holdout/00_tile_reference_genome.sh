# Reference data
GRCH38=input/GRCh38.chrom.sizes
BLACKLIST=input/ENCFF356LFX.bed.gz

# Script to tile the genome
TILE=../bin/tile_Genome.pl
SORT=../bin/sort_BED.pl

# Set size of tiles
SIZE=1000

# Split the genome into tiled chromosome files
perl $TILE $GRCH38 1000 GRCh38
cat *.bed > GRCh38_1000bp.bed

# Remove blacklist regions from genome to prevent their inclusion in training
bedtools intersect -v -a GRCh38_1000bp.bed -b $BLACKLIST > GRCh38_1000bp_filter.bed
perl $SORT GRCh38_1000bp_filter.bed GRCh38_1000bp_sort.bed
mv GRCh38_1000bp_sort.bed GRCh38_1000bp.bed

# Compress final file
gzip GRCh38_1000bp.bed

# Organize data
mkdir -p GRCh38_BED
mv *.gz GRCh38_BED/

# Cleanup
rm *.bed
