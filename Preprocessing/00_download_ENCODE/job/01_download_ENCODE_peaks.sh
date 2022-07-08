#ATAC-seq peaks
wget -O A549_ATAC_hg38_ENCFF899OMR.bed.gz https://www.encodeproject.org/files/ENCFF899OMR/@@download/ENCFF899OMR.bed.gz
wget -O HepG2_ATAC_hg38_ENCFF439EIO.bed.gz https://www.encodeproject.org/files/ENCFF439EIO/@@download/ENCFF439EIO.bed.gz
wget -O K562_ATAC_hg38_ENCFF333TAT.bed.gz https://www.encodeproject.org/files/ENCFF333TAT/@@download/ENCFF333TAT.bed.gz
wget -O MCF7_ATAC_hg38_ENCFF821OEF.bed.gz https://www.encodeproject.org/files/ENCFF821OEF/@@download/ENCFF821OEF.bed.gz

#Move datasets
mkdir -p ATAC
mv *.bed.gz ATAC/

#STARR-seq peaks
wget -O A549_STARR_hg38_ENCFF646OQS.bed.gz https://www.encodeproject.org/files/ENCFF646OQS/@@download/ENCFF646OQS.bed.gz
wget -O HepG2_STARR_hg38_ENCFF047LDJ.bed.gz https://www.encodeproject.org/files/ENCFF047LDJ/@@download/ENCFF047LDJ.bed.gz
wget -O K562_STARR_hg38_ENCFF045TVA.bed.gz https://www.encodeproject.org/files/ENCFF045TVA/@@download/ENCFF045TVA.bed.gz
wget -O MCF7_STARR_hg38_ENCFF826BPU.bed.gz https://www.encodeproject.org/files/ENCFF826BPU/@@download/ENCFF826BPU.bed.gz

#Move datasets
mkdir -p STARR
mv *.bed.gz STARR/
