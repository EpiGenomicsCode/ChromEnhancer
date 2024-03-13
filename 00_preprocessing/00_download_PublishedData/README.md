
This directory is for downloading processed datasets. This assumes you have [samtools](http://www.htslib.org/download/) installed.

## Download ENCODE ATAC-seq peaks
This downloads the ENCODE ATAC-seq peaks for A549, MCF7, K562, and HepG2

```
sh 01_download_ENCODE_peaks.sh
```

Moves peak BED files into `data/ATAC_Cellline`


## Download ENCODE ChIP-seq BAM files
This downloads ENCODE BAM files for H3K4me3, H3K27ac, CTCF, p300, and POL2RA in the A549, MCF7, K562, and HepG2 cell lines

```
sh 02_download_ENCODE_BAM.sh
```

Moves BAM files into `data/BAM`

#### Index BAM files
BAM files require a *.bai index file for efficient read-access

  ```
  cd ../data/BAM/
  for file in *.bam; do samtools index $file; done
  ```

## Download chromHMM genome annotations
This downloads hg38 chromHMM annotations from Kellis lab

```
sh 03_download_chromHMM.sh
```

Moves BAM files into `data/chromHMM_hg38`

## Download ENCODE ChIP-seq BAM files (extended set for figure generation)
This downloads ENCODE BAM files for H3K36me3, H3K27me3, H3K9me3, and H3K4me1 in the A549, MCF7, K562, and HepG2 cell lines

```
sh 04_download_ENCODE_BAM-extended.sh
```

Moves BAM files into `data/BAM`

## Download ENCODE ChIP-seq BAM files for Large Network training
This downloads 330 ENCODE ChIP-seq BAM files for the K562 cell line
 - Warning: This downloads over 600 GB worth of data

```
sh 05_download_ENCODE_BAM-large.sh
```

Moves BAM files into `data/BAM`
