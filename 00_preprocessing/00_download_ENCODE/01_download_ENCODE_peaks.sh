#ATAC-seq peaks
wget -O A549_ATAC_hg38_ENCFF899OMR.bed.gz https://www.encodeproject.org/files/ENCFF899OMR/@@download/ENCFF899OMR.bed.gz
wget -O HepG2_ATAC_hg38_ENCFF439EIO.bed.gz https://www.encodeproject.org/files/ENCFF439EIO/@@download/ENCFF439EIO.bed.gz
wget -O K562_ATAC_hg38_ENCFF333TAT.bed.gz https://www.encodeproject.org/files/ENCFF333TAT/@@download/ENCFF333TAT.bed.gz
wget -O MCF7_ATAC_hg38_ENCFF821OEF.bed.gz https://www.encodeproject.org/files/ENCFF821OEF/@@download/ENCFF821OEF.bed.gz

#Move datasets
mkdir -p ../../data/ATAC_Cellline
mv *.bed.gz ../../data/ATAC_Cellline/
