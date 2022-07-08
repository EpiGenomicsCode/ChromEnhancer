

This directory is for calling enhancer coordinates for training

## Call enhancers
Enhancers are called as the intersect of ATAC-seq peaks (in vivo accessible chromatin) and STARR-seq peaks (in vitro enhancer performance). Stringent enhancers are called using Starrpeaker peaks with a Q-value set to the default of 0.05. Stringent enhancer peaks are called using Starrpeaker peaks with a Q-value set to 0.1.

```
sh job/01_call_Enhancer.sh
```
