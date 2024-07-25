module load anaconda
CONVERT=../../bin/H5conversion/convert_CELL-chromtrack_to_h5.py

HOLDOUT=../../data/CELL-HOLDOUT/
TRAIN=../../data/CELL-TRAIN/

OUTTRAIN=../../data/CELL_NETWORK/TRAIN
mkdir -p $OUTTRAIN
OUTHOLD=../../data/CELL_NETWORK/HOLDOUT
mkdir -p $OUTHOLD

python $CONVERT --holdout_input=$HOLDOUT --holdout_output=$OUTHOLD --train_input=$TRAIN --train_output=$OUTTRAIN

