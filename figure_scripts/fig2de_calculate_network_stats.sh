#module load anaconda3_cpu/23.7.4
module load anaconda

# Establish folder structure
mkdir -p ../figures/fig2/paneld
mkdir -p ../figures/fig2/panele

PERF=../../../bin/chart/visualize_model-Stats_Violin.py

# Panel D
cd ../figures/fig2/paneld
grep "ROC" ../../../output-cell/results/*.txt > CELL_ROC.tab
python $PERF CELL_ROC.tab CELL_ROC.svg 0.75 1 

# Panel E
cd ../panele
grep "PRC" ../../../output-cell/results/*.txt > CELL_PRC.tab
python $PERF CELL_PRC.tab CELL_PRC.svg 0 0.4

