
# == Set-up Starrpeaker ==
#conda create -n starrpeaker python=2.7 pybedtools
#conda activate starrpeaker
#pip install git+https://github.com/gersteinlab/starrpeaker
#starrpeaker -h


# == Download Chrom.sizes ==
echo "Download GRCh38.chrom.sizes.simple.sorted..."
wget https://github.com/gersteinlab/starrpeaker/blob/master/data/GRCh38.chrom.sizes.simple.sorted
mv GRCh38.chrom.sizes.simple.sorted input/

#echo "Download hg38.chrom.sizes"
#wget -N https://hgdownload-test.gi.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
#mv hg38.chrom.sizes input/

# == Download Blacklist ==
echo "Download blacklist..."
wget https://www.encodeproject.org/files/ENCFF419RSJ/@@download/ENCFF419RSJ.bed.gz
gzip -d ENCFF419RSJ.bed.gz
mv ENCFF419RSJ.bed input/

# == Download Covariate files ==
echo "Download covariates (gc5Base.bw)"
wget https://hgdownload.soe.ucsc.edu/gbdb/hg38/bbi/gc5BaseBw/gc5Base.bw
mv gc5Base.bw input/

echo "Download covariates (mappability)"
echo "wget command goes here"
#mv input/

echo "Download covariates (RNA energy)"
echo "wget command goes here"
#mv input/


## Other files (not needed at this point)

# == Download Ref Genome ==
#echo "Download twoBitToFa"
#wget -N http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/twoBitToFa
#chmod 777 twoBitToFa
#mv twoBitToFa bin/

#echo "Download hg38 genome..."
#wget -N http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit
#echo "Converting 2bit to fa..."
#bin/twoBitToFa hg38.2bit input/hg38.fa

#echo "BWA Indexing genome..."
#bwa index input/hg38.fa
#echo "Complete"

# Clean up
#rm hg38.2bit

#echo "Download GRCh38 genome..."
#HG38=GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
#wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
#mv $HG38 input/
