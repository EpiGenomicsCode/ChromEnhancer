# set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

VENN=../../bin/chart/venn4py_CellLineParse.py

OUTPUT=../../figures/sfig2
[ -d $OUTPUT ] || mkdir -p $OUTPUT

# Enhancers - Stringent
A549=../../data/Enhancer_Coord/A549_hg38_StringentEnhancer_1000bp.bed.gz
HEPG2=../../data/Enhancer_Coord/HepG2_hg38_StringentEnhancer_1000bp.bed.gz
K562=../../data/Enhancer_Coord/K562_hg38_StringentEnhancer_1000bp.bed.gz
MCF7=../../data/Enhancer_Coord/MCF7_hg38_StringentEnhancer_1000bp.bed.gz

# A549 x K562
bedtools intersect -f 0.5 -wa    -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549;K562"}' > $OUTPUT/A549_K562.bed
bedtools intersect -f 0.5 -wa -v -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549"}' >> $OUTPUT/A549_K562.bed
bedtools intersect -f 0.5 -wa -v -a $K562  -b $A549  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"K562"}' >> $OUTPUT/A549_K562.bed

# MCF7 x HepG2
bedtools intersect -f 0.5 -wa    -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2;MCF7"}' > $OUTPUT/HepG2_MCF7.bed
bedtools intersect -f 0.5 -wa -v -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2"}' >> $OUTPUT/HepG2_MCF7.bed
bedtools intersect -f 0.5 -wa -v -a $MCF7  -b $HEPG2 | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"MCF7"}' >> $OUTPUT/HepG2_MCF7.bed

# Merge
bedtools intersect -f 0.5 -wo    -a $OUTPUT/A549_K562.bed  -b $OUTPUT/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4";"$8}' > $OUTPUT/Venn-Stringent.txt
bedtools intersect -f 0.5 -wa -v -a $OUTPUT/A549_K562.bed  -b $OUTPUT/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $OUTPUT/Venn-Stringent.txt
bedtools intersect -f 0.5 -wa -v -a $OUTPUT/HepG2_MCF7.bed -b $OUTPUT/A549_K562.bed  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $OUTPUT/Venn-Stringent.txt
	
# Make venn figure
python $VENN -i $OUTPUT/Venn-Stringent.txt -o $OUTPUT/
mv $OUTPUT/Venn_4.svg $OUTPUT/Venn-Stringent_4.svg

# Enhancers - Lenient
A549=../../data/Enhancer_Coord/A549_hg38_LenientEnhancer_1000bp.bed.gz
HEPG2=../../data/Enhancer_Coord/HepG2_hg38_LenientEnhancer_1000bp.bed.gz
K562=../../data/Enhancer_Coord/K562_hg38_LenientEnhancer_1000bp.bed.gz
MCF7=../../data/Enhancer_Coord/MCF7_hg38_LenientEnhancer_1000bp.bed.gz

# A549 x K562
bedtools intersect -f 0.5 -wa    -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549;K562"}' > $OUTPUT/A549_K562.bed
bedtools intersect -f 0.5 -wa -v -a $A549  -b $K562  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"A549"}' >> $OUTPUT/A549_K562.bed
bedtools intersect -f 0.5 -wa -v -a $K562  -b $A549  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"K562"}' >> $OUTPUT/A549_K562.bed

# MCF7 x HepG2
bedtools intersect -f 0.5 -wa    -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2;MCF7"}' > $OUTPUT/HepG2_MCF7.bed
bedtools intersect -f 0.5 -wa -v -a $HEPG2 -b $MCF7  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"HepG2"}' >> $OUTPUT/HepG2_MCF7.bed
bedtools intersect -f 0.5 -wa -v -a $MCF7  -b $HEPG2 | awk 'BEGIN{FS="\t";OFS="\t"}{print $1,$2,$3,"MCF7"}' >> $OUTPUT/HepG2_MCF7.bed

# Merge
bedtools intersect -f 0.5 -wo    -a $OUTPUT/A549_K562.bed  -b $OUTPUT/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4";"$8}' > $OUTPUT/Venn-Lenient.txt
bedtools intersect -f 0.5 -wa -v -a $OUTPUT/A549_K562.bed  -b $OUTPUT/HepG2_MCF7.bed | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $OUTPUT/Venn-Lenient.txt
bedtools intersect -f 0.5 -wa -v -a $OUTPUT/HepG2_MCF7.bed -b $OUTPUT/A549_K562.bed  | awk 'BEGIN{FS="\t";OFS="\t"}{print $1":"$2+500,$4}'       >> $OUTPUT/Venn-Lenient.txt
	
# Make venn figure
python $VENN -i $OUTPUT/Venn-Lenient.txt -o $OUTPUT/
mv $OUTPUT/Venn_4.svg $OUTPUT/Venn-Lenient_4.svg
rm $OUTPUT/*.bed $OUTPUT/*.txt
