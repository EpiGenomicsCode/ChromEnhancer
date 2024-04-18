module load anaconda3_cpu/23.7.4

# Establish folder structure
mkdir -p ../figures/fig2/panelb
mkdir -p ../figures/fig2/panelc

PERF=../../bin/visualize_model-Stats.py

# Panel B
cd ../figures/fig2/panelb
grep "ROC" ../../../output-chr/results/*.txt > CHR_ROC.tab
python $PERF CHR_ROC.tab CHR_ROC.svg

# Panel C
cd ../panelc
grep "PRC" ../../../output-chr/results/*.txt > CHR_PRC.tab
python $PERF CHR_PRC.tab CHR_PRC.svg
