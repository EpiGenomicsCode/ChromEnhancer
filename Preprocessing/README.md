## Overview
This folder contains the information required to completely regenerate the data files used in training and validating the EnhancerNN prediction algorithm.

All analysis is performed against the GRCh38 human reference genome.

## Final folder structure
When all scripts have been run sequentially, it will produce the following data files in this folder structure:

<details>
<summary> Preprocessing/00_download_ENCODE
</summary>

```
|--Preprocessing/00_download_ENCODE/ATAC
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
  |--README.md
  |--input
    |--bam-samples.txt
    |--ENCFF419RSJ.bed
    |--GRCh38.chrom.sizes.simple.sorted
    |--STARRPeaker_cov_GRCh38_gem-mappability-100mer.bw
    |--STARRPeaker_cov_GRCh38_linearfold-folding-energy-100bp.bw
    |--STARRPeaker_cov_GRCh38_ucsc-gc-5bp.bw
  |--job
    |--00_download_data.pbs
    |--01_run_starrpeaker.pbs
    |--setup.sh
  |--logs
    |--download.data.log.err-1
    |--download.data.log.err-2
    |--download.data.log.err-3
    |--download.data.log.err-4
    |--download.data.log.err-5
    |--download.data.log.err-6
    |--download.data.log.err-7
    |--download.data.log.err-8
    |--download.data.log.out-1
    |--download.data.log.out-2
    |--download.data.log.out-3
    |--download.data.log.out-4
    |--download.data.log.out-5
    |--download.data.log.out-6
    |--download.data.log.out-7
    |--download.data.log.out-8
    |--starrpeaker.log.err-1
    |--starrpeaker.log.err-2
    |--starrpeaker.log.err-3
    |--starrpeaker.log.err-4
    |--starrpeaker.log.out-1
    |--starrpeaker.log.out-2
    |--starrpeaker.log.out-3
    |--starrpeaker.log.out-4
  |--results
    |--BAM
      |--ENCFF060IFE.bam
      |--ENCFF060IFE.bam.bai
      |--ENCFF229JXT.bam
      |--ENCFF229JXT.bam.bai
      |--ENCFF287SIA.bam
      |--ENCFF287SIA.bam.bai
      |--ENCFF316KQD.bam
      |--ENCFF316KQD.bam.bai
      |--ENCFF323CMG.bam
      |--ENCFF323CMG.bam.bai
      |--ENCFF391WAQ.bam
      |--ENCFF391WAQ.bam.bai
      |--ENCFF427PXM.bam
      |--ENCFF427PXM.bam.bai
      |--ENCFF503CJW.bam
      |--ENCFF503CJW.bam.bai
      |--ENCFF655GXX.bam
      |--ENCFF655GXX.bam.bai
      |--ENCFF672URE.bam
      |--ENCFF672URE.bam.bai
      |--ENCFF807BAQ.bam
      |--ENCFF807BAQ.bam.bai
      |--ENCFF848IIW.bam
      |--ENCFF848IIW.bam.bai
    |--StarrpeakerResults
      |--A549.bam.bct
      |--A549.bin.bed
      |--A549.cov.tsv
      |--A549.fc.bw
      |--A549.input.bw
      |--A549.input.frag.bed
      |--A549.output.bw
      |--A549.output.frag.bed
      |--A549.peak.bed
*     |--A549.peak.final.bed
      |--A549.pval.bw
      |--A549.qval.bw
      |--HepG2.bam.bct
      |--HepG2.bin.bed
      |--HepG2.cov.tsv
      |--HepG2.fc.bw
      |--HepG2.input.bw
      |--HepG2.input.frag.bed
      |--HepG2.output.bw
      |--HepG2.output.frag.bed
      |--HepG2.peak.bed
*      |--HepG2.peak.final.bed
      |--HepG2.pval.bw
      |--HepG2.qval.bw
      |--K562.bam.bct
      |--K562.bin.bed
      |--K562.cov.tsv
      |--K562.fc.bw
      |--K562.input.bw
      |--K562.input.frag.bed
      |--K562.output.bw
      |--K562.output.frag.bed
      |--K562.peak.bed
*      |--K562.peak.final.bed
      |--K562.pval.bw
      |--K562.qval.bw
      |--MCF-7.bam.bct
      |--MCF-7.bin.bed
      |--MCF-7.cov.tsv
      |--MCF-7.fc.bw
      |--MCF-7.input.bw
      |--MCF-7.input.frag.bed
      |--MCF-7.output.bw
      |--MCF-7.output.frag.bed
      |--MCF-7.peak.bed
*      |--MCF-7.peak.final.bed
      |--MCF-7.pval.bw
      |--MCF-7.qval.bw
```

</details>

<br>

<details>
<summary> Preprocessing/02_call_Enhancers
</summary>

```
|--Preprocessing/02_call_Enhancers
  |--Enhancer_Coord
 |--A549_hg38_LenientEnhancer_1000bp.bed
  |--A549_hg38_LenientEnhancer.bed
    |--A549_hg38_StringentEnhancer_1000bp.bed
    |--A549_hg38_StringentEnhancer.bed
    |--HepG2_hg38_LenientEnhancer_1000bp.bed
    |--HepG2_hg38_LenientEnhancer.bed
    |--HepG2_hg38_StringentEnhancer_1000bp.bed
    |--HepG2_hg38_StringentEnhancer.bed
    |--K562_hg38_LenientEnhancer_1000bp.bed
    |--K562_hg38_LenientEnhancer.bed
    |--K562_hg38_StringentEnhancer_1000bp.bed
    |--K562_hg38_StringentEnhancer.bed
    |--MCF7_hg38_LenientEnhancer_1000bp.bed
    |--MCF7_hg38_LenientEnhancer.bed
    |--MCF7_hg38_StringentEnhancer_1000bp.bed
    |--MCF7_hg38_StringentEnhancer.bed
```

</details>

<br>

<details>
<summary> Preprocessing/03_generate_Training_data_CHR-holdout
</summary>

```
|--Preprocessing/03_generate_Training_data_CHR-holdout

```

</details>

<br>

<details>
<summary> Preprocessing/04_generate_Training_data_Cellline
</summary>

```
|--Preprocessing/04_generate_Training_data_Cellline

```

</details>
