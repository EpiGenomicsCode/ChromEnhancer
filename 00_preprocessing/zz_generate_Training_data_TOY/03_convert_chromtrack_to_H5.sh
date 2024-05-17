module load anaconda
CONVERT=../../bin/convert_TOY-chromtrack_to_h5.py

HOLDOUT=../../data/TOY-HOLDOUT/
TRAIN=../../data/TOY-TRAIN/

OUTTRAIN=../../data/TOY_NETWORK/TRAIN
mkdir -p $OUTTRAIN
OUTHOLD=../../data/TOY_NETWORK/HOLDOUT
mkdir -p $OUTHOLD

python $CONVERT --holdout_input=$HOLDOUT --holdout_output=$OUTHOLD --train_input=$TRAIN --train_output=$OUTTRAIN

