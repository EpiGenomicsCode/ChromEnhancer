module load anaconda3_cpu/23.7.4 

# Establish folder structure
mkdir -p ../figures/sfig2
cd ../figures/sfig2

LOSS=../../bin/visualize_Loss_StdDev.py 

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

