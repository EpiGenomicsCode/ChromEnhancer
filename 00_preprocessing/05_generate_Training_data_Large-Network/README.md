## Generate training and holdout data internal to a cell line
Running the scripts in order produces the training and holdout data used for the Large network training.

<details>
<summary> 00_preprocessing/05_generate_Training_data_Large-Network
</summary>

```
|--GRCh38_BED
  |--GRCh38_1000bp.bed.gz
|--HOLDOUT
  |--CELLLINE_LenientEnhancer_chrZ.bed
  |--CELLLINE_LenientEnhancer_chrZ.label
  |--CELLLINE_StringentEnhancer_chrZ.bed
  |--CELLLINE_StringentEnhancer_chrZ.label
  |--CELLINE_chrZ_CTCF-REP#_combined.chromtrack.gz
  |--CELLINE_chrZ_H3K27ac-REP#_combined.chromtrack.gz
  |--CELLINE_chrZ_H3K4me3-REP#_combined.chromtrack.gz
  |--CELLINE_chrZ_PolII-REP#_combined.chromtrack.gz
  |--CELLINE_chrZ_p300-REP#_combined.chromtrack.gz
|--TRAIN
  |--CELLLINE_chr10-chr17_train.bed
  |--CELLLINE_chr11-chr7_train.bed
  |--CELLLINE_chr12-chr8_train.bed
  |--CELLLINE_chr13-chr9_train.bed
  |--CELLLINE_chr14-chrX_train.bed
  |--CELLLINE_chr10-chr17_train.label
  |--CELLLINE_chr11-chr7_train.label
  |--CELLLINE_chr12-chr8_train.label
  |--CELLLINE_chr13-chr9_train.label
  |--CELLLINE_chr14-chrX_train.label
  |--CELLLINE_chrZ-chrA_train_CTCF-REP#_combined.chromtrack.gz
  |--CELLLINE_chrZ-chrA_train_H3K27ac-REP#_combined.chromtrack.gz
  |--CELLLINE_chrZ-chrA_train_H3K4me3-REP#_combined.chromtrack.gz
  |--CELLLINE_chrZ-chrA_train_PolII-REP#_combined.chromtrack.gz
  |--CELLLINE_chrZ-chrA_train_p300-REP#_combined.chromtrack.gz

```

</details>

<br>
