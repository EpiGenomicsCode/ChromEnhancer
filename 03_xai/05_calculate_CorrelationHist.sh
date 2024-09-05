set -exo
module load anaconda3_cpu

IDIR=../output-cell/xai
ODIR=../figures/fig5/paneld
[ -d $ODIR ] || mkdir -p $ODIR

# Scripts
BIN=bin/sum_columns.py
AVG=bin/average_columns.py
CORR=bin/calculate_correlation.py
HIST=bin/plot_histogram.py

INPUT=../output-cell/xai/CLD_study_-_test_-_valid_-_model1_clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-2_epoch_20.pt_xai_orig.tsv

# Create comparison TSV file
python $BIN $INPUT $ODIR/raw_data.tsv

for i in {1..10}; do

    python $AVG ../output-cell/swarm/iter$i/20_CLD_study_-_test_-_valid_-_model5_clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-2_epoch_20_particle_complessed.tsv $ODIR/iter$i\.tsv
    python $CORR $ODIR/raw_data.tsv $ODIR/iter$i\.tsv $ODIR/iter$i\_corr.tsv
    python $AVG ../output-cell/swarm_inverse/iter$i\/20_CLD_study_-_test_-_valid_-_model5_clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-2_epoch_20_particle_complessed.tsv $ODIR/iter$i\-neg.tsv
    python $CORR $ODIR/raw_data.tsv $ODIR/iter$i\-neg.tsv $ODIR/iter$i\-neg_corr.tsv
	
    python $HIST $ODIR/iter$i\_corr.tsv $ODIR/iter$i\-neg_corr.tsv $ODIR/iter$i\.svg

done
