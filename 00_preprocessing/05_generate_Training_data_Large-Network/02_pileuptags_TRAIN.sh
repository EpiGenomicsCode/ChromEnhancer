SCRIPTMANAGER=../../bin/ScriptManager-v0.14.jar
ALLBAM=../LARGE-BAM

TRAIN=$PWD\/../../data/LARGE-TRAIN
cd $TRAIN

JOBSTATS="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24GB
#SBATCH --time=6:00:00
#SBATCH --partition=open
cd $TRAIN"

for file in K562_*chr12*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

        CPU=0
        COUNTER=0
        sampleID=$1\_$COUNTER\.slurm
        rm -f $sampleID
        echo "$JOBSTATS" >> $sampleID

        for FACTOR in $ALLBAM/*.bam; do
                filePath="${FACTOR/.bam/}"
                BASE="${filePath##*/}"  # Removes everything before the last "/"
                fileID=$(echo "$BASE" | sed 's/^[^_]*_//')

                echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=4 --gzip --combined --output-matrix=$1\_$fileID $file $FACTOR" >> $sampleID

                # Process 6 BAM files per Slurm job
                let CPU++
                if [[ $CPU -eq 6 ]]; then
                        wait
                        CPU=0
                        let COUNTER++
                        sampleID=$1\_$COUNTER\.slurm
                        rm -f $sampleID
                        echo "$JOBSTATS" >> $sampleID
                fi
        done


done

# Submit jobs to cluster
for file in *.slurm; do sbatch $file; done
