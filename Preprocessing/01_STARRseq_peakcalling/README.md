

This directory is for downloading ENCODE datasets


## Set-up
Install Starrpeaker according to the [documentation](https://github.com/gersteinlab/starrpeaker).

```
conda create -n starrpeaker python=2.7 pybedtools
conda activate starrpeaker
pip install git+https://github.com/gersteinlab/starrpeaker
starrpeaker -h
```

## Download Reference Files
Get the GRCh38 genome build-specific reference files for running the Starrpeaker on the datasets.
`bash job/setup.sh`

The covariate files will need to be manually downloaded from the starrpeaker repository's [Google Drive links](https://github.com/gersteinlab/starrpeaker#covariates) and moved to the `input/` directory. When everything is setup, the input directory should contain these files:
```
|--input
  |--bam-samples.txt
  |--ENCFF419RSJ.bed
  |--GRCh38.chrom.sizes.simple.sorted
  |--STARRPeaker_cov_GRCh38_gem-mappability-100mer.bw
  |--STARRPeaker_cov_GRCh38_linearfold-folding-energy-100bp.bw
  |--STARRPeaker_cov_GRCh38_ucsc-gc-5bp.bw
```

## Download datasets from ENCODE
ENCODE file codes are pulled and uniqued from `input/bam-samples.txt` in the third column and the sixth column (STARRseq-input control pairs). These include the merged replicates of STARR-seq data for K564, A549, HepG2, and MCF-7 with their input controls (2 files, shared between the 4 samples).

Update the `WRK=` path in the `job/00_download_data.pbs` submission script. Also update the configurations of the PBS submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria).

`qsub job/00_download_data.pbs`

This should write 6 BAM-formatted and indexed files to a new directory `results/BAM/` (input controls are shared between certain samples). STDERR and STDOUT written to files in `logs`.


## Call putative enhancer peaks with Starrpeaker
Get a set of peak annotations using Starrpeaker that will identify enhancers within each cell line (output named with appropriate prefix that includes the cell line).

Update the `WRK=` path in the `job/01_run_starrpeaker.pbs` submission script. Also update the configurations of the submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria).

`qsub job/01_run_starrpeaker.pbs`

This should write 4 sets of Starrpeaker output to a new directory `results/StarrpeakerResults/`. STDERR and STDOUT written to files in `logs`.
