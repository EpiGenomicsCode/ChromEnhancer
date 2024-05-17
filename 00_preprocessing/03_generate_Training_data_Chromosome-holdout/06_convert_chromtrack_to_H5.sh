module load anaconda
CONVERT=../../bin/convert_CHR-chromtrack_to_h5.py

HOLDOUT=../../data/CHR-HOLDOUT/
TRAIN=../../data/CHR-TRAIN/

OUTTRAIN=../../data/CHR_NETWORK/TRAIN
mkdir -p $OUTTRAIN
OUTHOLD=../../data/CHR_NETWORK/HOLDOUT
mkdir -p $OUTHOLD

python $CONVERT --holdout_input=$HOLDOUT --holdout_output=$OUTHOLD --train_input=$TRAIN --train_output=$OUTTRAIN

