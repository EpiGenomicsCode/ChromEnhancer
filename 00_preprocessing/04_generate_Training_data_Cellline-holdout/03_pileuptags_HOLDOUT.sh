SCRIPTMANAGER=../../bin/ScriptManager-v0.14.jar
FACTORBAM=../BAM
HISTONEBAM=../BAM

HOLDOUT=$PWD\/../../data/CELL-HOLDOUT
cd $HOLDOUT

JOBSTATS="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=24GB
#SBATCH --time=6:00:00
#SBATCH --partition=open
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
H3K4ME1_1=$HISTONEBAM/K562_H3K4me1_ENCFF580LGK.bam
H3K4ME1_2=$HISTONEBAM/K562_H3K4me1_ENCFF778EZR.bam
H3K36ME3_1=$HISTONEBAM/K562_H3K36me3_ENCFF594GRL.bam
H3K36ME3_2=$HISTONEBAM/K562_H3K36me3_ENCFF925FDY.bam
H3K27ME3_1=$HISTONEBAM/K562_H3K27me3_ENCFF392ZKG.bam
H3K27ME3_2=$HISTONEBAM/K562_H3K27me3_ENCFF905CZD.bam

for file in K562_enhancer*.bed.gz; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.slurm
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-1 $file $H3K4ME1_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-2 $file $H3K4ME1_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-1 $file $H3K36ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-2 $file $H3K36ME3_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-1 $file $H3K27ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-2 $file $H3K27ME3_2" >> $sampleID

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
H3K4ME1_1=$HISTONEBAM/A549_H3K4me1_ENCFF189DIW.bam
H3K4ME1_2=$HISTONEBAM/A549_H3K4me1_ENCFF843JEO.bam
H3K36ME3_1=$HISTONEBAM/A549_H3K36me3_ENCFF347QGE.bam
H3K36ME3_2=$HISTONEBAM/A549_H3K36me3_ENCFF701IUT.bam
H3K27ME3_1=$HISTONEBAM/A549_H3K27me3_ENCFF168ZZS.bam
H3K27ME3_2=$HISTONEBAM/A549_H3K27me3_ENCFF747ZKE.bam

for file in A549_enhancer*.bed.gz; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.slurm
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=100 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=85 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-1 $file $H3K4ME1_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-2 $file $H3K4ME1_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-1 $file $H3K36ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-2 $file $H3K36ME3_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-1 $file $H3K27ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-2 $file $H3K27ME3_2" >> $sampleID

done

CTCF_1=$FACTORBAM/HepG2_CTCF_ENCFF012FMD.bam
CTCF_2=$FACTORBAM/HepG2_CTCF_ENCFF487UUI.bam
P300_1=$FACTORBAM/HepG2_p300_ENCFF352YDX.bam
P300_2=$FACTORBAM/HepG2_p300_ENCFF953FZD.bam
POL_1=$FACTORBAM/HepG2_POLR2A_ENCFF835GBL.bam
POL_2=$FACTORBAM/HepG2_POLR2A_ENCFF845YGC.bam
H3K4ME3_1=$HISTONEBAM/HepG2_H3K4me3_ENCFF426UGV.bam
H3K4ME3_2=$HISTONEBAM/HepG2_H3K4me3_ENCFF223DKE.bam
H3K27AC_1=$HISTONEBAM/HepG2_H3K27ac_ENCFF686HFQ.bam
H3K27AC_2=$HISTONEBAM/HepG2_H3K27ac_ENCFF805KGN.bam
H3K4ME1_1=$HISTONEBAM/HepG2_H3K4me1_ENCFF256HMH.bam
H3K4ME1_2=$HISTONEBAM/HepG2_H3K4me1_ENCFF372VZP.bam
H3K36ME3_1=$HISTONEBAM/HepG2_H3K36me3_ENCFF080RGC.bam
H3K36ME3_2=$HISTONEBAM/HepG2_H3K36me3_ENCFF211APO.bam
H3K27ME3_1=$HISTONEBAM/HepG2_H3K27me3_ENCFF027OKJ.bam
H3K27ME3_2=$HISTONEBAM/HepG2_H3K27me3_ENCFF369DOB.bam

for file in HepG2_enhancer*.bed.gz; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.slurm
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=85 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-1 $file $H3K4ME1_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-2 $file $H3K4ME1_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-1 $file $H3K36ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-2 $file $H3K36ME3_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-1 $file $H3K27ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-2 $file $H3K27ME3_2" >> $sampleID

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
H3K4ME1_1=$HISTONEBAM/MCF7_H3K4me1_ENCFF592EVS.bam
H3K4ME1_2=$HISTONEBAM/MCF7_H3K4me1_ENCFF748ISL.bam
H3K36ME3_1=$HISTONEBAM/MCF7_H3K36me3_ENCFF551PNK.bam
H3K36ME3_2=$HISTONEBAM/MCF7_H3K36me3_ENCFF747AWB.bam
H3K27ME3_1=$HISTONEBAM/MCF7_H3K27me3_ENCFF413QYQ.bam
H3K27ME3_2=$HISTONEBAM/MCF7_H3K27me3_ENCFF744JQU.bam

for file in MCF7_enhancer*.bed.gz; do
	var=$(echo $file | awk -F"." '{print $1}')
	set -- $var
	echo $1

	sampleID=$1\_CTCF-1\.slurm
	rm -f $sampleID
	echo "$JOBSTATS" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-1 $file $CTCF_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=50 --output-matrix=$1\_CTCF-2 $file $CTCF_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_p300-1 $file $P300_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=40 --output-matrix=$1\_p300-2 $file $P300_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_PolII-1 $file $POL_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=75 --output-matrix=$1\_PolII-2 $file $POL_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-1 $file $H3K4ME3_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K4me3-2 $file $H3K4ME3_2" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K27ac-1 $file $H3K27AC_1" >> $sampleID
#	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=90 --output-matrix=$1\_H3K27ac-2 $file $H3K27AC_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-1 $file $H3K4ME1_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=110 --output-matrix=$1\_H3K4me1-2 $file $H3K4ME1_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-1 $file $H3K36ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K36me3-2 $file $H3K36ME3_2" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-1 $file $H3K27ME3_1" >> $sampleID
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip --combined --shift=80 --output-matrix=$1\_H3K27me3-2 $file $H3K27ME3_2" >> $sampleID

done

# Submit jobs to cluster
for file in *.slurm; do sbatch $file; done
