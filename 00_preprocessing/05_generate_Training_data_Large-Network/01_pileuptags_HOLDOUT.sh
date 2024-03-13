SCRIPTMANAGER=../../bin/ScriptManager-v0.14.jar
ALLBAM=../BAM

HOLDOUT=$PWD\/../../data/LARGE-HOLDOUT
cd $HOLDOUT

JOBSTATS="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24GB
#SBATCH --time=6:00:00
#SBATCH --partition=open
cd $HOLDOUT"

for file in K562_Stringent*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	CPU=0

	for FACTOR in $ALLBAM/*.bam; do
	        fileID="${FACTOR/.bam/}"
		echo $1,"\t",$fileID

        	# Multi-thread to 8 cores
	        let CPU++
	        if [[ $CPU -eq 8 ]]; then
	                wait
	                CPU=0
	        fi
	done
done
wait
exit



	sampleID=$1\.slurm
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID

done

# Submit jobs to cluster
#for file in *.slurm; do sbatch $file; done
