#!/bin/bash
#PBS -l nodes=1:ppn=6
#PBS -l pmem=24gb
#PBS -l walltime=05:00:00
#PBS -A open
#PBS -o logs/download.encode.peaks.log.out
#PBS -e logs/download.encode.peaks.log.err
#PBS -t 1-2786

WRK=/path/to/Enhanced_Transformer_For_Enhancers/Validation
WRK=/storage/home/owl5022/scratch/Validation

cd $WRK

[ -d logs ] || mkdir logs
[ -d results/PEAKS ] || mkdir -p results/PEAKS

for PBS_ARRAYID in {1..2786};
do

INDEX=$((PBS_ARRAYID+1))

METADATA=AllPeaks_A549-MCF7-HepG2-K562.tab
INFO=`sed "${INDEX}q;d" $METADATA`
ENCFF=`echo $INFO | awk '{print $1}'`
ASSAY=`echo $INFO | awk '{print $5}'`
CELL=`echo $INFO | awk '{print $3}'`
TARGET=`echo $INFO | awk '{print $4}'`
#echo $INFO

OFILE=results/PEAKS/$ENCFF\_$ASSAY\_$CELL\_$TARGET\_peaks.bed.gz

# ENCODE data download
HREF=/files/$ENCFF/@@download/$ENCFF.bed.gz
echo "Fetching from https://www.encodeproject.org$HREF"
wget -c -O $OFILE https://www.encodeproject.org$HREF

# # Checksum of resulting BAM
# MD5SUM=`echo $INFO | awk '{print $3}'`
# if [[ `md5sum $BAM` =~ $MD5SUM ]]; then
# 	echo "($PBS_ARRAYID) $BAM passed."
# else
# 	echo "($PBS_ARRAYID) $BAM md5checksum failed!"
# 	rm $BAM
# 	exit
# fi

done