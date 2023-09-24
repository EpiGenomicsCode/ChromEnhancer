
# Set-up

## Dependencies
```
conda create -n everything -c bioconda -c conda-forge pandas pysam scipy samtools bedtools seaborn fast-histogram colorcet datashader scikit-learn
```

## Downloads

ENCODE peaks from all TFs for K562, A549, HepG2, and MCF-7.

Cell Line-specific TF Expression from Haining:
|        | A549 | HepG2 | K562 | MCF7 |
| ------ | ---- | ----- | ---- | ---- |
| GRHL2  |      |       |      |  X   |
| HNF4A  |      |  X    |      |      |
| ESR1   |      |       |      |  X   |
| FOXA1  |  X   |  X    |      |  X   |
| FOXA2  |  X   |  X    |      |      |
| FOXC1  |  X   |  X    |      |      |
| GATA1  |      |       |  X   |      |
| MEIS2  |      |       |  X   |      |
| STAT5A |      |       |  X   |      |
| SPi1   |      |       |      |      |
| JunB   |      |       |      |      |
| ZFAT   |      |       |  X   |      |
| Sox2   |      |       |      |      |
| EHMT2  |      |       |  X   |      |
| NFYC   |      |       |      |      |
| BCL6   |      |       |      |      |
| KLF4   |      |  X    |      |      |
| FOS    |      |       |      |  X   |
| GATA3  |      |       |      |  X   |
| PBX1   |      |       |      |  X   |



```
python get_ENCFF-Peak_from_ENCODEsearch.py \
	-i "https://www.encodeproject.org/search/?type=File&file_format_type=narrowPeak&file_type=bed+narrowPeak&assay_title=TF+ChIP-seq&biosample_ontology.term_name=K562&biosample_ontology.term_name=A549&biosample_ontology.term_name=MCF-7&biosample_ontology.term_name=HepG2&assembly=GRCh38&status=released&output_type=conservative+IDR+thresholded+peaks&format=json&limit=all" \
	-o AllPeaks_A549-MCF7-HepG2-K562.tab

```

info needed
- target
- assay
- file type
- genome build
- cell line
- audit info
- quality metrics
- encsr
- replicate

# 1 Aggregate Enhancer Scores
```
qsub job/01_aggrergate_scores_and_call_enhancers.pbs
```

```
results/ResortByName/A549_Rep1_EnhancerScore_SORT.bed
results/ResortByName/A549_Rep2_EnhancerScore_SORT.bed
results/ResortByName/HepG2_Rep1_EnhancerScore_SORT.bed
results/ResortByName/HepG2_Rep2_EnhancerScore_SORT.bed
results/ResortByName/K562_Rep1_EnhancerScore_SORT.bed
results/ResortByName/K562_Rep2_EnhancerScore_SORT.bed
results/ResortByName/MCF7_Rep1_EnhancerScore_SORT.bed
results/ResortByName/MCF7_Rep2_EnhancerScore_SORT.bed
```

## For each genomic tile, aggregate
- Score value matrix
- 0/1 calls by threshold = 0.5
- 0/1 calls by threshold = 0.9

|   | K562_1 | K562_2 | A549_1 | ... |
| - | ------ | ------ | ------ | --- |
| Tile1 | 1  | 1 | 0 | ... |
| Tile2 | 0  | 0 | 1 | ... |
| Tile3 | 0  | 0 | 0 | ... |

```
results/ALL_SCORES.out
results/Enhancers_T-50.out
results/Enhancers_T-90.out
```

## Filter to just enhancers
- Filter rows without a single enhancer "call" (only 0/1 called matrices)

```
results/JustEnhancers_T-50.out
results/JustEnhancers_T-90.out
```

## Merge Replicate columns
Create versions with the following merge criteria:
- union
- intersection

```
results/MergeReplicates/
```

# 1 Aggregate ENCODE Peaks



# Score Distributions
- Assess consistency between replicates sigmoid score (rep1/rep2 scatter)
- Histogram each dataset sigmoid score
	- For each: is there a natural cutoff between bound/unbound?
	- For each: where does the top 10K line sit?
	- is the shape generally the same/cutoff the same between cell lines

# Set analysis
- Repeat top 10k v threshold
- Pairwise matrix of relationships:
	- Intersect
	- Union
	- Jaccard similarity index or related
- Hierarchical clustering of matrix


# TF analysis
- Aggregate ENCODE peaks
- Pileups of TF peaks from ENCODE
	- Sort by hierarchical groups --> Peak align --> Aggregate data --> Heatmap --> look for associations