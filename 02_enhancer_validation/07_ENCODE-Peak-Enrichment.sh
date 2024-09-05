#!/bin/bash
#SBATCH -A bbse-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=24:00:00

# Calculate ENCODE ChIP peak enrichment at enhancer calls from model N
#   1. Intersect ENCODE peak files with enhancer tiles to get contingency table counts of genomic tile overlaps
#   2. Perform chi-square test for independence (between peak calls and enhancer calls) with Benjamini-Hochberg multiple test correction for each row of contingency table values
#   3. Reshape the corrected p-value statistics into a table with TF target per row and cell line per column for heatmap visualization
#   4. Perform hierarchical clustering on independence p-values

# |--TF-BoundMetrics.txt
# |--PoissonMetrics.txt
# |--PoissonMetrics_filter.txt
# |--PoissonMetrics_filter.svg

set -exo
module load anaconda3_cpu
source activate /scratch/bbse/wklai/EnhancerNN/bedtools

ODIR=../tables/table1/PeakEnrichment
[ -d $ODIR ] || mkdir -p $ODIR

# Outputs
BOUNDMETRICS=$ODIR/TF-BoundMetrics.txt

# Script shortcuts
POISSON=bin/poisson_means_test_from_tf-bound.py
RESHAPE=bin/reshape_PoissonStats.py 
HEATMAP=bin/generate_TF-Heatmap.py

# Merge cell line stats to single file
cat $ODIR/TF-BoundMetrics_A549.txt > $ODIR/TF-BoundMetrics.txt
tail -n +2 $ODIR/TF-BoundMetrics_HepG2.txt >> $ODIR/TF-BoundMetrics.txt
tail -n +2 $ODIR/TF-BoundMetrics_K562.txt >> $ODIR/TF-BoundMetrics.txt
tail -n +2 $ODIR/TF-BoundMetrics_MCF7.txt >> $ODIR/TF-BoundMetrics.txt

# Check signficance (per row basis)
python $POISSON -i $BOUNDMETRICS -o $ODIR/PoissonMetrics.txt

# Reorganize by CL - reshape table by CL into matrix
python $RESHAPE $ODIR/PoissonMetrics.txt $ODIR/PoissonMetrics_filter.txt

# Heatmap
python $HEATMAP $ODIR/PoissonMetrics_filter.txt $ODIR/PoissonMetrics_filter.svg
