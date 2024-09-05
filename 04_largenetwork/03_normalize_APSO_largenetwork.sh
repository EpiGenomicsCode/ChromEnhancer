set -exo
module load anaconda3_cpu

IDIR=../output-large/swarm
ODIR=../figures/fig6/panelb
[ -d $ODIR ] || mkdir -p $ODIR

FACTOR=input/330_FactorID.txt

SCALE=bin/normalize_APSO_by_Grad.py
AVG=bin/average_columns.py
SORT=bin/sort_Factor_by_Score.py

for i in {1..10}; do

	python $SCALE $IDIR/iter$i\/50_LargeDataset1_6_epoch_100_particle_complessed.tsv $IDIR/iter$i\/50_LargeDataset1_6_epoch_100_grad_compressed.tsv $ODIR/iter$i\_matrix.tsv
	python $AVG $ODIR/iter$i\_matrix.tsv $ODIR/iter$i\_avg.tsv
	python $SORT $FACTOR $ODIR/iter$i\_avg.tsv $ODIR/iter$i\_factor.tsv

done

rm $ODIR/*avg.tsv $ODIR/*matrix.tsv
