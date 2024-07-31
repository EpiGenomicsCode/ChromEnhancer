# # ROAR Collab
# module load anaconda
# Delta
module load anaconda3_cpu

PERF=../../../bin/chart/visualize_model-Stats_Violin.py

# Panel D

# Establish folder structure
FIG2D=../../figures/fig2/paneld
[ -d $FIG2D ] || mkdir -p $FIG2D

cd $FIG2D
grep "ROC" ../../../output-cell/results/*.txt > CELL_ROC.tab
python $PERF CELL_ROC.tab CELL_ROC.svg 0.75 1 

# Panel E

# Establish folder structure
FIG2E=../panele
[ -d $FIG2E ] || mkdir -p $FIG2E

cd $FIG2E
grep "PRC" ../../../output-cell/results/*.txt > CELL_PRC.tab
python $PERF CELL_PRC.tab CELL_PRC.svg 0 0.4

