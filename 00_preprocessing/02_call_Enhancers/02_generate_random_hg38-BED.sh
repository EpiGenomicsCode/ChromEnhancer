module load bedtools

GENOME=input/GRCh38.chrom.sizes
BLACKLIST=input/ENCFF356LFX.bed.gz
CHRY=input/hg38_chrY.bed

# Generate 30K random hg38 genomics coordinates of size 1kb
bedtools random -l 1000 -n 30000 -seed 10 -g $GENOME > temp.bed
# Remove random sites that intersect with blacklist regions
bedtools intersect -v -a temp.bed -b $BLACKLIST > temp2.bed
# Remove random sites that intersect with chrY
bedtools intersect -v -a temp2.bed -b $CHRY > temp.bed

# Get final 25K random genomic coordinates
head -n 25000 temp.bed > hg38_25KRand_1000bp.bed

# Clean up temp files
rm temp.bed temp2.bed

# Organize data
gzip *.bed
mkdir -p ../../data/Random_Coord
mv *.bed.gz ../../data/Random_Coord/
