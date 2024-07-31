# # ROAR Collab
# module load anaconda
# Delta
module load anaconda3_cpu

# Establish folder structure
SFIG2=../../figures/sfig2
[ -d $SFIG2 ] || mkdir -p $SFIG2

LOSS=../../bin/chart/visualize_Loss_StdDev.py 

# Chromosome Holdout
python $LOSS ../../output-chr/loss/ model1 model1_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model2 model2_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model3 model3_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model4 model4_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model5 model5_chr-holdout_Loss.svg
python $LOSS ../../output-chr/loss/ model6 model6_chr-holdout_Loss.svg

# Cell Holdout
python $LOSS ../../output-cell/loss/ model1 model1_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model2 model2_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model3 model3_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model4 model4_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model5 model5_cell-holdout_Loss.svg
python $LOSS ../../output-cell/loss/ model6 model6_cell-holdout_Loss.svg

