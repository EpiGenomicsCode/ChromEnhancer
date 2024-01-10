## Generate training and holdout data across a cell line
Running the scripts in order produces the training and holdout data used for the chromsomal-holdout procedure used to validate the network.

<details>
<summary> Preprocessing/04_generate_Training_data_Cellline
</summary>

```
|--GRCh38_BED
  |--GRCh38_1000bp.bed.gz
|--HOLDOUT
  |--CELLLINE_LenientEnhancer.bed.gz
  |--CELLLINE_LenientEnhancer.label
  |--CELLLINE_StringentEnhancer.bed.gz
  |--CELLLINE_StringentEnhancer.label
  |--CELLINE_CTCF-REP#_combined.chromtrack.gz
  |--CELLINE_H3K27ac-REP#_combined.chromtrack.gz
  |--CELLINE_H3K4me3-REP#_combined.chromtrack.gz
  |--CELLINE_PolII-REP#_combined.chromtrack.gz
  |--CELLINE_p300-REP#_combined.chromtrack.gz
|--TRAIN
  |--CELLLINE_train.bed
  |--CELLLINE_train.label
  |--CELLLINE_train_CTCF-REP#_combined.chromtrack.gz
  |--CELLLINE_train_H3K27ac-REP#_combined.chromtrack.gz
  |--CELLLINE_train_H3K4me3-REP#_combined.chromtrack.gz
  |--CELLLINE_train_PolII-REP#_combined.chromtrack.gz
  |--CELLLINE_train_p300-REP#_combined.chromtrack.gz

```

</details>

<br>
