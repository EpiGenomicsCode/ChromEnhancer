module load anaconda3_cpu/23.7.4
#module load anaconda

# Establish folder structure
mkdir -p ../figures/fig2/panelb
mkdir -p ../figures/fig2/panelc

PERF=../../../bin/chart/visualize_model-Stats_Violin.py

# Panel B
cd ../figures/fig2/panelb
grep "ROC" ../../../output-chr/results/*.txt > CHR_ROC.tab
python $PERF CHR_ROC.tab CHR_ROC.svg 0.75 1

# Panel C
cd ../panelc
grep "PRC" ../../../output-chr/results/*.txt > CHR_PRC.tab
python $PERF CHR_PRC.tab CHR_PRC.svg 0 0.4
