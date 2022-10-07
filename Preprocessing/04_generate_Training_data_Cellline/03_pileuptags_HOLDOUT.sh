SCRIPTMANAGER=../../Preprocessing/bin/ScriptManager-v0.13-dev.jar

FACTORBAM=../data/BAM
HISTONEBAM=../data/BAM

HOLDOUT=HOLDOUT
cd $HOLDOUT

JOBSTATS="#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l pmem=24gb
#PBS -l walltime=2:00:00
#PBS -A open
cd $HOLDOUT"

CTCF_1=$FACTORBAM/K562_CTCF_ENCFF198CVB.bam
CTCF_2=$FACTORBAM/K562_CTCF_ENCFF488CXC.bam
P300_1=$FACTORBAM/K562_p300_ENCFF200PYZ.bam
P300_2=$FACTORBAM/K562_p300_ENCFF982AFE.bam
POL_1=$FACTORBAM/K562_POLR2A_ENCFF201SIE.bam
POL_2=$FACTORBAM/K562_POLR2A_ENCFF267TTN.bam
H3K4ME3_1=$HISTONEBAM/K562_H3K4me3_ENCFF236SNL.bam
H3K4ME3_2=$HISTONEBAM/K562_H3K4me3_ENCFF661UGK.bam
H3K27AC_1=$HISTONEBAM/K562_H3K27ac_ENCFF301TVL.bam
H3K27AC_2=$HISTONEBAM/K562_H3K27ac_ENCFF879BWC.bam

for file in K562*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.pbs
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID

done

CTCF_1=$FACTORBAM/A549_CTCF_ENCFF280TYK.bam
CTCF_2=$FACTORBAM/A549_CTCF_ENCFF835YDD.bam
P300_1=$FACTORBAM/A549_p300_ENCFF040EMK.bam
P300_2=$FACTORBAM/A549_p300_ENCFF138AMX.bam
POL_1=$FACTORBAM/A549_POLR2A_ENCFF641ZJE.bam
POL_2=$FACTORBAM/A549_POLR2A_ENCFF816DKP.bam
H3K4ME3_1=$HISTONEBAM/A549_H3K4me3_ENCFF428UWO.bam
H3K4ME3_2=$HISTONEBAM/A549_H3K4me3_ENCFF643FMK.bam
H3K27AC_1=$HISTONEBAM/A549_H3K27ac_ENCFF117TAC.bam
H3K27AC_2=$HISTONEBAM/A549_H3K27ac_ENCFF273YZW.bam

for file in A549*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.pbs
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=85 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID

done

CTCF_1=$FACTORBAM/HepG2_CTCF_ENCFF012FMD.bam
CTCF_2=$FACTORBAM/HepG2_CTCF_ENCFF487UUI.bam
P300_1=$FACTORBAM/HepG2_p300_ENCFF352YDX.bam
P300_2=$FACTORBAM/HepG2_p300_ENCFF953FZD.bam
POL_1=$FACTORBAM/HepG2_POLR2A_ENCFF835GBL.bam
POL_2=$FACTORBAM/HepG2_POLR2A_ENCFF845YGC.bam
H3K4ME3_1=$HISTONEBAM/HepG2_H3K4me3_ENCFF060PGB.bam
H3K4ME3_2=$HISTONEBAM/HepG2_H3K4me3_ENCFF360OCU.bam
H3K27AC_1=$HISTONEBAM/HepG2_H3K27ac_ENCFF686HFQ.bam
H3K27AC_2=$HISTONEBAM/HepG2_H3K27ac_ENCFF805KGN.bam

for file in HepG2*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.pbs
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=85 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID

done

CTCF_1=$FACTORBAM/MCF7_POLR2A_ENCFF191BDN.bam
CTCF_2=$FACTORBAM/MCF7_POLR2A_ENCFF193BNK.bam
P300_1=$FACTORBAM/MCF7_CTCF_ENCFF049OXC.bam
P300_2=$FACTORBAM/MCF7_CTCF_ENCFF959AJO.bam
POL_1=$FACTORBAM/MCF7_p300_ENCFF359OVO.bam
POL_2=$FACTORBAM/MCF7_p300_ENCFF596FSA.bam
H3K4ME3_1=$HISTONEBAM/MCF7_H3K4me3_ENCFF371XST.bam
H3K4ME3_2=$HISTONEBAM/MCF7_H3K4me3_ENCFF716OCC.bam
H3K27AC_1=$HISTONEBAM/MCF7_H3K27ac_ENCFF096GIM.bam
H3K27AC_2=$HISTONEBAM/MCF7_H3K27ac_ENCFF692SZU.bam

for file in MCF7*.bed; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.pbs
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID

done

# Submit jobs to cluster
for file in *.pbs; do qsub $file; done
