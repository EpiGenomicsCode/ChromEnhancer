## Overview
This folder contains the information required to completely regenerate the data files used in training and validating the EnhancerNN prediction algorithm.

All analysis is performed against the GRCh38 human reference genome.

## Final folder structure
When all scripts have been run sequentially, it will produce the following data files in this folder structure:

<details>
<summary> Preprocessing/00_download_ENCODE
</summary>

```
|--Preprocessing/00_download_ENCODE
  |--ATAC
    |--A549_ATAC_hg38_ENCFF899OMR.bed.gz
    |--HepG2_ATAC_hg38_ENCFF439EIO.bed.gz
    |--K562_ATAC_hg38_ENCFF333TAT.bed.gz
    |--MCF7_ATAC_hg38_ENCFF821OEF.bed.gz
  |--STARR
    |--A549_STARR_hg38_ENCFF646OQS.bed.gz
    |--HepG2_STARR_hg38_ENCFF047LDJ.bed.gz
    |--K562_STARR_hg38_ENCFF045TVA.bed.gz
    |--MCF7_STARR_hg38_ENCFF826BPU.bed.gz
  |--BAM
    |--K562_H3K4me3_ENCFF236SNL.bam
    |--K562_H3K4me3_ENCFF661UGK.bam
    |--K562_H3K27ac_ENCFF301TVL.bam
    |--K562_H3K27ac_ENCFF879BWC.bam
    |--K562_CTCF_ENCFF198CVB.bam
    |--K562_CTCF_ENCFF488CXC.bam
    |--K562_p300_ENCFF200PYZ.bam
    |--K562_p300_ENCFF982AFE.bam
    |--K562_POLR2A_ENCFF201SIE.bam
    |--K562_POLR2A_ENCFF267TTN.bam
    |--A549_H3K4me3_ENCFF973TUQ.bam
    |--A549_H3K4me3_ENCFF643FMK.bam
    |--A549_H3K4me3_ENCFF428UWO.bam
    |--A549_H3K27ac_ENCFF393XCS.bam
    |--A549_H3K27ac_ENCFF117TAC.bam
    |--A549_H3K27ac_ENCFF273YZW.bam
    |--A549_CTCF_ENCFF280TYK.bam
    |--A549_CTCF_ENCFF835YDD.bam
    |--A549_p300_ENCFF040EMK.bam
    |--A549_p300_ENCFF138AMX.bam
    |--A549_POLR2A_ENCFF641ZJE.bam
    |--A549_POLR2A_ENCFF816DKP.bam
    |--HepG2_H3K4me3_ENCFF360OCU.bam
    |--HepG2_H3K4me3_ENCFF060PGB.bam
    |--HepG2_H3K27ac_ENCFF805KGN.bam
    |--HepG2_H3K27ac_ENCFF686HFQ.bam
    |--HepG2_CTCF_ENCFF012FMD.bam
    |--HepG2_CTCF_ENCFF487UUI.bam
    |--HepG2_p300_ENCFF352YDX.bam
    |--HepG2_p300_ENCFF953FZD.bam
    |--HepG2_POLR2A_ENCFF835GBL.bam
    |--HepG2_POLR2A_ENCFF845YGC.bam
    |--MCF7_H3K4me3_ENCFF716OCC.bam
    |--MCF7_H3K4me3_ENCFF371XST.bam
    |--MCF7_H3K27ac_ENCFF692SZU.bam
    |--MCF7_H3K27ac_ENCFF096GIM.bam
    |--MCF7_CTCF_ENCFF049OXC.bam
    |--MCF7_CTCF_ENCFF959AJO.bam
    |--MCF7_p300_ENCFF359OVO.bam
    |--MCF7_p300_ENCFF596FSA.bam
    |--MCF7_POLR2A_ENCFF191BDN.bam
    |--MCF7_POLR2A_ENCFF193BNK.bam
    |--K562_H3K4me3_ENCFF236SNL.bam.bai
    |--K562_H3K4me3_ENCFF661UGK.bam.bai
    |--K562_H3K27ac_ENCFF301TVL.bam.bai
    |--K562_H3K27ac_ENCFF879BWC.bam.bai
    |--K562_CTCF_ENCFF198CVB.bam.bai
    |--K562_CTCF_ENCFF488CXC.bam.bai
    |--K562_p300_ENCFF200PYZ.bam.bai
    |--K562_p300_ENCFF982AFE.bam.bai
    |--K562_POLR2A_ENCFF201SIE.bam.bai
    |--K562_POLR2A_ENCFF267TTN.bam.bai
    |--A549_H3K4me3_ENCFF973TUQ.bam.bai
    |--A549_H3K4me3_ENCFF643FMK.bam.bai
    |--A549_H3K4me3_ENCFF428UWO.bam.bai
    |--A549_H3K27ac_ENCFF393XCS.bam.bai
    |--A549_H3K27ac_ENCFF117TAC.bam.bai
    |--A549_H3K27ac_ENCFF273YZW.bam.bai
    |--A549_CTCF_ENCFF280TYK.bam.bai
    |--A549_CTCF_ENCFF835YDD.bam.bai
    |--A549_p300_ENCFF040EMK.bam.bai
    |--A549_p300_ENCFF138AMX.bam.bai
    |--A549_POLR2A_ENCFF641ZJE.bam.bai
    |--A549_POLR2A_ENCFF816DKP.bam.bai
    |--HepG2_H3K4me3_ENCFF360OCU.bam.bai
    |--HepG2_H3K4me3_ENCFF060PGB.bam.bai
    |--HepG2_H3K27ac_ENCFF805KGN.bam.bai
    |--HepG2_H3K27ac_ENCFF686HFQ.bam.bai
    |--HepG2_CTCF_ENCFF012FMD.bam.bai
    |--HepG2_CTCF_ENCFF487UUI.bam.bai
    |--HepG2_p300_ENCFF352YDX.bam.bai
    |--HepG2_p300_ENCFF953FZD.bam.bai
    |--HepG2_POLR2A_ENCFF835GBL.bam.bai
    |--HepG2_POLR2A_ENCFF845YGC.bam.bai
    |--MCF7_H3K4me3_ENCFF716OCC.bam.bai
    |--MCF7_H3K4me3_ENCFF371XST.bam.bai
    |--MCF7_H3K27ac_ENCFF692SZU.bam.bai
    |--MCF7_H3K27ac_ENCFF096GIM.bam.bai
    |--MCF7_CTCF_ENCFF049OXC.bam.bai
    |--MCF7_CTCF_ENCFF959AJO.bam.bai
    |--MCF7_p300_ENCFF359OVO.bam.bai
    |--MCF7_p300_ENCFF596FSA.bam.bai
    |--MCF7_POLR2A_ENCFF191BDN.bam.bai
    |--MCF7_POLR2A_ENCFF193BNK.bam.bai
```

</details>

<br>

<details>
<summary> Preprocessing/01_STARRseq_peakcalling
</summary>

```
|--Preprocessing/01_STARRseq_peakcalling

```

</details>

<br>

<details>
<summary> Preprocessing/02_call_Enhancers
</summary>

```
|--Preprocessing/02_call_Enhancers
  |--Enhancer_Coord
    |--A549_hg38_LenientEnhancer_1000bp.bed.gz
    |--A549_hg38_LenientEnhancer.bed.gz
    |--A549_hg38_StringentEnhancer_1000bp.bed.gz
    |--A549_hg38_StringentEnhancer.bed.gz
    |--HepG2_hg38_LenientEnhancer_1000bp.bed.gz
    |--HepG2_hg38_LenientEnhancer.bed.gz
    |--HepG2_hg38_StringentEnhancer_1000bp.bed.gz
    |--HepG2_hg38_StringentEnhancer.bed.gz
    |--K562_hg38_LenientEnhancer_1000bp.bed.gz
    |--K562_hg38_LenientEnhancer.bed.gz
    |--K562_hg38_StringentEnhancer_1000bp.bed.gz
    |--K562_hg38_StringentEnhancer.bed.gz
    |--MCF7_hg38_LenientEnhancer_1000bp.bed.gz
    |--MCF7_hg38_LenientEnhancer.bed.gz
    |--MCF7_hg38_StringentEnhancer_1000bp.bed.gz
    |--MCF7_hg38_StringentEnhancer.bed.gz
```

</details>

<br>

<details>
<summary> Preprocessing/03_generate_Training_data_CHR-holdout
</summary>

```
|--Preprocessing/03_generate_Training_data_CHR-holdout
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

<details>
<summary> Preprocessing/04_generate_Training_data_Cellline
</summary>

```
|--Preprocessing/04_generate_Training_data_Cellline
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
