
# ROAR Collab
module load anaconda
module load bedtools

# # Delta
# module load anaconda3_cpu
# source activate /scratch/bbse/wklai/EnhancerNN/bedtools/

FIG1=../../figures/sfig1
[ -d $SFIG1 ] || mkdir -p $SFIG1
cd $SFIG1

# chromHMM
A549=../../data/chromHMM_hg38/A549_BSS00007_18_CALLS_segments.bed.gz
HEPG2=../../data/chromHMM_hg38/HepG2_BSS00558_18_CALLS_segments.bed.gz
K562=../../data/chromHMM_hg38/K562_BSS00762_18_CALLS_segments.bed.gz
MCF7=../../data/chromHMM_hg38/MCF7_BSS01226_18_CALLS_segments.bed.gz

# Enhancers - Stringent
A549_E=../../data/Enhancer_Coord/A549_hg38_StringentEnhancer_1000bp.bed.gz
HEPG2_E=../../data/Enhancer_Coord/HepG2_hg38_StringentEnhancer_1000bp.bed.gz
K562_E=../../data/Enhancer_Coord/K562_hg38_StringentEnhancer_1000bp.bed.gz
MCF7_E=../../data/Enhancer_Coord/MCF7_hg38_StringentEnhancer_1000bp.bed.gz

bedtools intersect -u -f 0.5 -a $A549 -b $A549_E > A549_Stringent-chromHMM.tab
bedtools intersect -u -f 0.5 -a $HEPG2 -b $HEPG2_E > HepG2_Stringent-chromHMM.tab
bedtools intersect -u -f 0.5 -a $K562 -b $K562_E > K562_Stringent-chromHMM.tab
bedtools intersect -u -f 0.5 -a $MCF7 -b $MCF7_E > MCF7_Stringent-chromHMM.tab

# Enhancers - Lenient
A549_E=../../data/Enhancer_Coord/A549_hg38_LenientEnhancer_1000bp.bed.gz
HEPG2_E=../../data/Enhancer_Coord/HepG2_hg38_LenientEnhancer_1000bp.bed.gz
K562_E=../../data/Enhancer_Coord/K562_hg38_LenientEnhancer_1000bp.bed.gz
MCF7_E=../../data/Enhancer_Coord/MCF7_hg38_LenientEnhancer_1000bp.bed.gz

bedtools intersect -u -f 0.5 -a $A549 -b $A549_E > A549_Lenient-chromHMM.tab
bedtools intersect -u -f 0.5 -a $HEPG2 -b $HEPG2_E > HepG2_Lenient-chromHMM.tab
bedtools intersect -u -f 0.5 -a $K562 -b $K562_E > K562_Lenient-chromHMM.tab
bedtools intersect -u -f 0.5 -a $MCF7 -b $MCF7_E > MCF7_Lenient-chromHMM.tab

# Calculate frequency
CALC=../../bin/calculate_chromHMM-occurence.pl
perl $CALC A549_Stringent-chromHMM.tab A549_Stringent-chromHMM_FREQ.tab
perl $CALC HepG2_Stringent-chromHMM.tab HepG2_Stringent-chromHMM_FREQ.tab
perl $CALC K562_Stringent-chromHMM.tab K562_Stringent-chromHMM_FREQ.tab
perl $CALC MCF7_Stringent-chromHMM.tab MCF7_Stringent-chromHMM_FREQ.tab

perl $CALC A549_Lenient-chromHMM.tab A549_Lenient-chromHMM_FREQ.tab
perl $CALC HepG2_Lenient-chromHMM.tab HepG2_Lenient-chromHMM_FREQ.tab
perl $CALC K562_Lenient-chromHMM.tab K562_Lenient-chromHMM_FREQ.tab
perl $CALC MCF7_Lenient-chromHMM.tab MCF7_Lenient-chromHMM_FREQ.tab

# Generate figure
GEN=../../bin/generate_chromHMM-Barchart.py
python $GEN A549_Stringent-chromHMM_FREQ.tab HepG2_Stringent-chromHMM_FREQ.tab K562_Stringent-chromHMM_FREQ.tab MCF7_Stringent-chromHMM_FREQ.tab Stringent_Enhancer-chromHMM.svg Stringent_Enhancer-chromHMM_Legend.svg
python $GEN A549_Lenient-chromHMM_FREQ.tab HepG2_Lenient-chromHMM_FREQ.tab K562_Lenient-chromHMM_FREQ.tab MCF7_Lenient-chromHMM_FREQ.tab Lenient_Enhancer-chromHMM.svg Lenient_Enhancer-chromHMM_Legend.svg
