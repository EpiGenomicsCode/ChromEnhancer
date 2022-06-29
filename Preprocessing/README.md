

This directory is for processing ENCODE datasets into 


## Set-up 
Install Starrpeaker according to the [documentation]:(https://github.com/gersteinlab/starrpeaker). 

```
conda create -n starrpeaker python=2.7 pybedtools
conda activate starrpeaker
pip install git+https://github.com/gersteinlab/starrpeaker
starrpeaker -h
```

## Download Reference Files
Get the GRCh38 genome build-specific reference files for running the Starrpeaker on the datasets.
`bash job/setup.sh`


## Download datasets from ENCODE
ENCODE file codes are pulled and uniqued from `input/bam-samples.txt` in the third column and the sixth column (STARRseq-input control pairs). These include the merged replicates of STARR-seq data for K564, A549, HepG2, and MCF-7 with their input controls.

Update the `WRK=` path in the `job/00_download_data.pbs` submission script. Also update the configurations of the submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria).

`qsub job/00_download_data.pbs`

This should write 6 BAM-formatted and indexed files to a new directory `results/BAM/` (input controls are shared between certain samples). STDERR and STDOUT written to files in `logs`.


## Call Enhancer Peaks with Starrpeaker
Get a set of peak annotations using Starrpeaker that will identify enhancers within each cell line (output named with appropriate prefix that includes the cell line).

Update the `WRK=` path in the `job/00_download_data.pbs` submission script. Also update the configurations of the submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria).

`qsub job/01_run_starrpeaker.pbs`

This should write 4 sets of Starrpeaker output to a new directory `results/StarrpeakerResults/`. STDERR and STDOUT written to files in `logs`.


## Process data into appropriate chromtrack format (unfinished)
The peak annotations will be processed into the appropriate data folder in a format that Jamil can input and process for the NN.

This means creating the various HOLDOUT-TRAIN chunk combinations:
```
Data/ENCODEdata
|--HOLDOUT
  |--CLNAME_enhancer_chr7.bed
  |--CLNAME_enhancer_chr8.bed
  |--CLNAME_enhancer_chr9.bed
  |--CLNAME_enhancer_chr10.bed
  |--CLNAME_enhancer_chr11.bed
  |--CLNAME_enhancer_chr12.bed
  |--CLNAME_enhancer_chr13.bed
  |--CLNAME_enhancer_chr14.bed
  |--CLNAME_enhancer_chr15.bed
  |--CLNAME_enhancer_chr16.bed
  |--CLNAME_enhancer_chr17.bed
  |--CLNAME_enhancer_chrX.bed
|--TRAIN
  |--TRAIN/CLNAME_chr10-chr17_train.bed
  |--TRAIN/CLNAME_chr11-chr7_train.bed
  |--TRAIN/CLNAME_chr12-chr8_train.bed
  |--TRAIN/CLNAME_chr13-chr9_train.bed
  |--TRAIN/CLNAME_chr14-chrX_train.bed
```

`qsub job/02_format_data.pbs (TODO)`

