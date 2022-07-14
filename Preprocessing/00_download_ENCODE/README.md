
This directory is for downloading processed ENCODE datasets. This assumes you have [samtools](http://www.htslib.org/download/) installed

## Download ENCODE peaks files
This downloads the ENCODE ATAC-seq and STARR-seq peaks files for A549, MCF7, K562, and HepG2

```
sh job/01_download_ENCODE_peaks.sh
```

## Download ENCODE chromatin BAM files
This downloads ENCODE BAM files for H3K4me3, H3K27ac, CTCF, p300, and POL2RA in the A549, MCF7, K562, and HepG2 cell lines

```
sh job/02_download_ENCODE_BAM.sh
```

## Organize BAM files
Move BAM files into `Preprocessing/00_download_ENCODE/BAM`

```
mkdir -p BAM
mv *.bam BAM/
```

## Index BAM files
BAM files require a *.bai index file for efficient read-access

```
cd BAM/
for file in *.bam; do samtools index $file; done
```
