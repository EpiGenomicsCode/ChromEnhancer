# # ROAR Collab
# module load anaconda
# Delta
module load anaconda3_cpu

# Establish folder structure
SFIG=../../figures/sfig3
[ -d $SFIG ] || mkdir -p $SFIG

LOSS=../../bin/chart/visualize_Loss_StdDev.py 

# Chromosome Holdout
python $LOSS ../../output-chr/loss/ model1 $SFIG/model1_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model2 $SFIG/model2_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model3 $SFIG/model3_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model4 $SFIG/model4_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model5 $SFIG/model5_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model6 $SFIG/model6_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model7 $SFIG/model7_chr-holdout_Loss.svg

# Cell Holdout
python $LOSS ../../output-cell/loss/ model1 $SFIG/model1_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model2 $SFIG/model2_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model3 $SFIG/model3_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model4 $SFIG/model4_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model5 $SFIG/model5_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model6 $SFIG/model6_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model7 $SFIG/model7_cell-holdout_Loss.svg
