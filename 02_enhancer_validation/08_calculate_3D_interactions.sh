set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

MODEL=model5

FILTER=bin/filter_ChIA-links.py
BARCHART=bin/generate_BarChart.py

GENE=input/knownCanonical_1000bp.bed
RAND=../data/Random_Coord/hg38_25KRand_1000bp.bed.gz

ODIR=../figures/fig3/panelc
[ -d $ODIR ] || mkdir -p $ODIR

# # Get data from ENCODE portal
# # K562 POLR2A
# wget https://www.encodeproject.org/files/ENCFF511QFN/@@download/ENCFF511QFN.bedpe.gz
# # K562 CTCF
# wget https://www.encodeproject.org/files/ENCFF118PBQ/@@download/ENCFF118PBQ.bedpe.gz
# # HepG2 POLR2A
# wget https://www.encodeproject.org/files/ENCFF364UNM/@@download/ENCFF364UNM.bedpe.gz
# # HepG2 CTCF
# wget https://www.encodeproject.org/files/ENCFF299NHM/@@download/ENCFF299NHM.bedpe.gz
# # MCF-7 POLR2A
# wget https://www.encodeproject.org/files/ENCFF597SQA/@@download/ENCFF597SQA.bedpe.gz
# # MCF-7 CTCF
# wget https://www.encodeproject.org/files/ENCFF633JWS/@@download/ENCFF633JWS.bedpe.gz
# # A549 POLR2A
# wget https://www.encodeproject.org/files/ENCFF421KYP/@@download/ENCFF421KYP.bedpe.gz
# # A549 CTCF
# wget https://www.encodeproject.org/files/ENCFF790WGN/@@download/ENCFF790WGN.bedpe.gz
# gunzip *.bedpe.gz
# mkdir -p ../data/ChIAPET
# mv *.bedpe ../data/ChIAPET/

CELL="A549"
ENH=../figures/fig3/panelb/$MODEL/Intersect/$MODEL\_$CELL\.bed
COUNT=$(wc -l < $ENH)
gunzip -c $RAND | head -n $COUNT > RAND.bed

TARGET="POLR2A"
CHIA=../data/ChIAPET/ENCFF421KYP.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

TARGET="CTCF"
CHIA=../data/ChIAPET/ENCFF790WGN.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

CELL="K562"
ENH=../figures/fig3/panelb/$MODEL/Intersect/$MODEL\_$CELL\.bed
COUNT=$(wc -l < $ENH)
gunzip -c $RAND | head -n $COUNT > RAND.bed

TARGET="POLR2A"
CHIA=../data/ChIAPET/ENCFF511QFN.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

TARGET="CTCF"
CHIA=../data/ChIAPET/ENCFF118PBQ.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

CELL="HepG2"
ENH=../figures/fig3/panelb/$MODEL/Intersect/$MODEL\_$CELL\.bed
COUNT=$(wc -l < $ENH)
gunzip -c $RAND | head -n $COUNT > RAND.bed

TARGET="POLR2A"
CHIA=../data/ChIAPET/ENCFF364UNM.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

TARGET="CTCF"
CHIA=../data/ChIAPET/ENCFF299NHM.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

CELL="MCF7"
ENH=../figures/fig3/panelb/$MODEL/Intersect/$MODEL\_$CELL\.bed
COUNT=$(wc -l < $ENH)
gunzip -c $RAND | head -n $COUNT > RAND.bed

TARGET="POLR2A"
CHIA=../data/ChIAPET/ENCFF597SQA.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

TARGET="CTCF"
CHIA=../data/ChIAPET/ENCFF633JWS.bedpe
cut -d$'\t' -f 1,2,3 $CHIA > ChIA_Side1.bed
cut -d$'\t' -f 4,5,6 $CHIA > ChIA_Side2.bed

bedtools intersect -c -a ChIA_Side1.bed -b $GENE > ChIA_Side1-GeneInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $GENE > ChIA_Side2-GeneInt.tsv
bedtools intersect -c -a ChIA_Side1.bed -b $ENH > ChIA_Side1-EnhInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b $ENH > ChIA_Side2-EnhInt.tsv

paste ChIA_Side1-GeneInt.tsv ChIA_Side1-EnhInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-EnhInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Enh.tsv

bedtools intersect -c -a ChIA_Side1.bed -b RAND.bed > ChIA_Side1-RandInt.tsv
bedtools intersect -c -a ChIA_Side2.bed -b RAND.bed > ChIA_Side2-RandInt.tsv
paste ChIA_Side1-GeneInt.tsv ChIA_Side1-RandInt.tsv ChIA_Side2-GeneInt.tsv ChIA_Side2-RandInt.tsv > temp.tsv
python $FILTER temp.tsv ChIA_$CELL\_$TARGET\-Rand.tsv

rm *-GeneInt.tsv *-EnhInt.tsv *-RandInt.tsv *Side*.bed temp.tsv RAND.bed

# Output data and make barchart
mv *tsv $ODIR
python $BARCHART $ODIR $ODIR/barchart.svg
