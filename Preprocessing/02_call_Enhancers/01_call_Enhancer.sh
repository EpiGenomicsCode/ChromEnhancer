# ENCODE ATAC seq peak file path
ATAC=ATAC_Cellline

# STARR-seq peak file path
STARR=STARR_Cellline

for file in "$ATAC"/*bed.gz; do
	ID="$(basename $file)"
	# Sort ATAC-seq file
	gunzip -c $file | sort -k 1,1 -k2,2n - > SORT.bed
	# Merge pseudo-replicated peaks to single peak
	bedtools merge -i SORT.bed > MERGE.bed
	# Get cell line ID from file name
	CELL=(`echo $ID | tr '_' ' '`)
	# Intersect ATAC-seq peaks with stringent STARR-seq peaks
	bedtools intersect -u -a MERGE.bed -b $STARR/$CELL\_pval0.05*.bed.gz > $CELL\_hg38_StringentEnhancer.bed
        # Intersect ATAC-seq peaks with lenient STARR-seq peaks
        bedtools intersect -u -a MERGE.bed -b $STARR/$CELL\_pval0.1*.bed.gz > $CELL\_hg38_LenientEnhancer.bed

done
# Clean up
rm SORT.bed MERGE.bed

# Resize all enhancers to X bp window
RESIZE=job/expand_BED.pl
SIZE=1000

for file in *Enhancer.bed; do
        temp="${file/.bed/_999bp.bed}"
        newFile="${temp/999/$SIZE}"
        echo $newFile
        perl $RESIZE $file $SIZE $newFile
done

# Organize data
gzip *.bed
mkdir -p Enhancer_Coord
mv *.bed.gz Enhancer_Coord/
