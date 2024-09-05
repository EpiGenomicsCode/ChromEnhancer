# # ROAR Collab
# module load anaconda
# Delta
module load anaconda3_cpu

PERF=../../../bin/chart/visualize_model-Stats_Violin.py

# Panel A

# Establish folder structure
FIG=../../figures/fig6/panela
[ -d $FIG ] || mkdir -p $FIG

cd $FIG
grep "ROC" ../../../output-large/results/*.txt > CELL_ROC.tab
python $PERF CELL_ROC.tab CELL_ROC.svg 0.5 1 

# Panel B

# Establish folder structure
FIG=../panelb
[ -d $FIG ] || mkdir -p $FIG

cd $FIG
grep "PRC" ../../../output-large/results/*.txt > CELL_PRC.tab
python $PERF CELL_PRC.tab CELL_PRC.svg 0 0.6

