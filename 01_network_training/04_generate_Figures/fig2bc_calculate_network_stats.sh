# # ROAR Collab
# module load anaconda
# Delta
module load anaconda3_cpu

PERF=../../../bin/chart/visualize_model-Stats_Violin.py

# Panel B

# Establish folder structure
FIG2B=../../figures/fig2/panelb
[ -d $FIG2B ] || mkdir -p $FIG2B
cd $FIG2B

grep "ROC" ../../../output-chr/results/*.txt > CHR_ROC.tab
python $PERF CHR_ROC.tab CHR_ROC.svg 0.75 1

# Panel C

# Establish folder structure
FIG2C=../panelc
[ -d $FIG2C ] || mkdir -p $FIG2C
cd $FIG2C

grep "PRC" ../../../output-chr/results/*.txt > CHR_PRC.tab
python $PERF CHR_PRC.tab CHR_PRC.svg 0 0.4
