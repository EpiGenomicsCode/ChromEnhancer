

This directory is for calling less stringent peaks from STARR-seq data.

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


## Call putative enhancer peaks with Starrpeaker
Get a set of peak annotations using Starrpeaker that will identify enhancers within each cell line (output named with appropriate prefix that includes the cell line).

Update the `WRK=` path in the `job/00_download_data.pbs` submission script. Also update the configurations of the submission to use the appropriate allocation name (and optionally adjust memory and cpu to fit your allocation criteria).

`qsub job/01_run_starrpeaker.pbs`

This should write 4 sets of Starrpeaker output to a new directory `results/StarrpeakerResults/`. STDERR and STDOUT written to files in `logs`.
