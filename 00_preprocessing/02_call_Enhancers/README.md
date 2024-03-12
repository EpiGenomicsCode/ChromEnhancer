

This directory is for calling enhancer coordinates for training. This assumes you have [bedtools](http://www.htslib.org/download/) installed.

## Call enhancers
Enhancers are called as the intersect of ATAC-seq peaks (in vivo accessible chromatin) and STARR-seq peaks (in vitro enhancer performance). Stringent enhancers are called using Starrpeaker peaks with a Q-value set to the default of 0.05. Lenient enhancer peaks are called using Starrpeaker peaks with a Q-value set to 0.1.

```
sh 01_call_Enhancer.sh
```

Moves directory into `data/Enhancer_Coord`

```
mv Enhancer_Coord ../../data/Enhancer_Coord
```

## Establish random peaks for background

```
sh 02_generate_random_hg38-BED.sh
```

Moves directory into `data/Random_Coord`

```
mv Enhancer_Coord ../../data/Random_Coord
```
