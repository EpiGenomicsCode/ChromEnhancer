# ENCODE ATAC seq peak file path
ATAC=/storage/home/wkl2/wkl2/Enhancer-NN_Project/00_DATA/ATAC_Cellline

# Stringent enhancers with Starrpeaker at Q0.05
STARR=/storage/home/wkl2/wkl2/Enhancer-NN_Project/00_DATA/STARR_Cellline

bedtools intersect -u -a $ATAC/A549_ATAC_hg38_ENCFF899OMR.bed.gz -b $STARR/A549_STARR_hg38_ENCFF646OQS.bed.gz > A549_hg38_StringentEnhancer.bed
bedtools intersect -u -a $ATAC/HepG2_ATAC_hg38_ENCFF439EIO.bed.gz -b $STARR/HepG2_STARR_hg38_ENCFF047LDJ.bed.gz > HepG2_hg38_StringentEnhancer.bed
bedtools intersect -u -a $ATAC/K562_ATAC_hg38_ENCFF333TAT.bed.gz -b $STARR/K562_STARR_hg38_ENCFF045TVA.bed.gz > K562_hg38_StringentEnhancer.bed
bedtools intersect -u -a $ATAC/MCF7_ATAC_hg38_ENCFF821OEF.bed.gz -b $STARR/MCF7_STARR_hg38_ENCFF826BPU.bed.gz > MCF7_hg38_StringentEnhancer.bed

# Lenient enhancers with Starrpeaker at Q0.1
STARR=/storage/home/wkl2/wkl2/Enhancer-NN_Project/00_DATA/STARR_Cellline

bedtools intersect -u -a $ATAC/A549_ATAC_hg38_ENCFF899OMR.bed.gz -b $STARR/A549_STARR_hg38_ENCFF646OQS.bed.gz > A549_hg38_LenientEnhancer.bed
bedtools intersect -u -a $ATAC/HepG2_ATAC_hg38_ENCFF439EIO.bed.gz -b $STARR/HepG2_STARR_hg38_ENCFF047LDJ.bed.gz > HepG2_hg38_LenientEnhancer.bed
bedtools intersect -u -a $ATAC/K562_ATAC_hg38_ENCFF333TAT.bed.gz -b $STARR/K562_STARR_hg38_ENCFF045TVA.bed.gz > K562_hg38_LenientEnhancer.bed
bedtools intersect -u -a $ATAC/MCF7_ATAC_hg38_ENCFF821OEF.bed.gz -b $STARR/MCF7_STARR_hg38_ENCFF826BPU.bed.gz > MCF7_hg38_LenientEnhancer.bed

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
mkdir -p Enhancer_Coord
mv *.bed Enhancer_Coord/
