module load anaconda
CONVERT=../../bin/H5conversion/convert_LARGE-chromtrack_to_h5.py 

HOLDOUT=../../data/LARGE-HOLDOUT/
TRAIN=../../data/LARGE-TRAIN/

OUTTRAIN=../../data/LARGE_NETWORK/TRAIN
mkdir -p $OUTTRAIN
OUTHOLD=../../data/LARGE_NETWORK/HOLDOUT
mkdir -p $OUTHOLD

python $CONVERT --holdout_input=$HOLDOUT --holdout_output=$OUTHOLD --train_input=$TRAIN --train_output=$OUTTRAIN

