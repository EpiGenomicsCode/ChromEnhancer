

This directory is for calling less stringent peaks from STARR-seq data. This assumes you have [wget](https://www.gnu.org/software/wget/) installed.

## Set-up Starrpeaker
Install Starrpeaker according to the [documentation](https://github.com/gersteinlab/starrpeaker).

```
conda create -n starrpeaker python=2.7 pybedtools
conda activate starrpeaker
pip install git+https://github.com/gersteinlab/starrpeaker
starrpeaker -h
```

## Download Reference Files
Get the GRCh38 genome build-specific reference files for running the Starrpeaker on the datasets and move them into `Preprocessing/01_STARRseq_peakcalling/input`

### GRCh38.chrom.sizes.simple.sorted (chrom.sizes)
```
wget https://github.com/gersteinlab/starrpeaker/blob/master/data/GRCh38.chrom.sizes.simple.sorted
mv GRCh38.chrom.sizes.simple.sorted input/
```

### ENCFF419RSJ.bed (blacklist)
```
wget https://www.encodeproject.org/files/ENCFF419RSJ/@@download/ENCFF419RSJ.bed.gz
gzip -d ENCFF419RSJ.bed.gz
mv ENCFF419RSJ.bed input/
```

### Covariates
Files need to be downloaded from the browser using Google Drive [links in the Starrpeaker docs](https://github.com/gersteinlab/starrpeaker#covariates).
1. GC content - `STARRPeaker_cov_GRCh38_ucsc-gc-5bp.bw`
2. Mappability - `STARRPeaker_cov_GRCh38_gem-mappability-100mer.bw`
3. RNA energy - `STARRPeaker_cov_GRCh38_linearfold-folding-energy-100bp.bw`

## Call putative enhancer peaks with Starrpeaker
Get a set of peak annotations using Starrpeaker that will identify enhancers within each cell line (output named with appropriate prefix that includes the cell line).

Update the `WRK=` path in the `job/01_run_starrpeaker.pbs` PBS submission script. Also update the configurations of the submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria). These commands take a long time to run so local sequential execution is not recommended.

`qsub job/01_run_starrpeaker.pbs`

This should write 4 sets of Starrpeaker output to a new directory `Preprocessing/data/STARR_Cellline/`. STDERR and STDOUT written to files in `logs`.
