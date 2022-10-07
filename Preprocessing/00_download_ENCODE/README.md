
This directory is for downloading processed ENCODE datasets. This assumes you have [samtools](http://www.htslib.org/download/) installed.

## Download ENCODE peaks files
This downloads the ENCODE ATAC-seq and STARR-seq peaks files for A549, MCF7, K562, and HepG2

```
sh 01_download_ENCODE_peaks.sh
```

## Organize Peak files
Move peak BED files into `Preprocessing/data/ATAC_Cellline` and `Preprocessing/data/STARR_Cellline`

```
mv ATAC_Cellline ../data/
mv STARR_Cellline ../data/
```

Note: ENCODE's STARR_Cellline peak files were called very stringently. The next preprocessing script generates more lenient peaks which are used moving forward.


## Download ENCODE chromatin BAM files
This downloads ENCODE BAM files for H3K4me3, H3K27ac, CTCF, p300, and POL2RA in the A549, MCF7, K562, and HepG2 cell lines

```
sh 02_download_ENCODE_BAM.sh
```

## Organize BAM files
Move BAM files into `Preprocessing/data/BAM`

```
mv BAM ../data/
```

## Index BAM files
BAM files require a *.bai index file for efficient read-access

```
cd ../data/BAM/
for file in *.bam; do samtools index $file; done
```
