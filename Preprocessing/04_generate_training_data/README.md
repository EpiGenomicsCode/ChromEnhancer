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
