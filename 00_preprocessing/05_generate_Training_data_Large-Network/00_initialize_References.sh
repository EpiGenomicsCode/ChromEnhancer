# Make Large-network reference folders
mkdir -p ../../data/LARGE-HOLDOUT
mkdir -p ../../data/LARGE-TRAIN

# Copy K562 datasets from Chromosome-holdout into Large-network for re-use
cp ../../data/CHR-HOLDOUT/K562*bed ../../data/LARGE-HOLDOUT/
cp ../../data/CHR-HOLDOUT/K562*label ../../data/LARGE-HOLDOUT/

cp ../../data/CHR-TRAIN/K562*bed ../../data/LARGE-TRAIN/
cp ../../data/CHR-TRAIN/K562*label ../../data/LARGE-TRAIN/

