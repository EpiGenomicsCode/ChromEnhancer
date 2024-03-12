mkdir -p ../figures/fig1/panelb
cd ../figures/fig1/panelb

# Panel B
SCRIPTMANAGER=../../../bin/ScriptManager-v0.14.jar
JOBPATH=$PWD\/../../../figures/fig1/panelb

JOBSTATS="#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=24GB
#SBATCH --time=1:00:00
#SBATCH --partition=open
cd $JOBPATH"

ENHANCER=../../../data/Enhancer_Coord/K562_hg38_StringentEnhancer_1000bp.bed.gz
sampleID=K562_Stringent.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/K562*bam; do
	# Extract the base filename using basename
	filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
	echo "$filename"
	echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Stringent $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=K562_PROseq_SRP045616-Stringent $ENHANCER ../../../data/PRO-BAM/K562_PROseq_SRP045616.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/K562_hg38_LenientEnhancer_1000bp.bed.gz
sampleID=K562_Lenient.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/K562*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Lenient $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=K562_PROseq_SRP045616-Lenient $ENHANCER ../../../data/PRO-BAM/K562_PROseq_SRP045616.bam" >> $sampleID

ENHANCER=../../../data/Random_Coord/hg38_25KRand_1000bp.bed.gz
sampleID=K562_Random.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/K562*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Random $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=K562_PROseq_SRP045616-Random $ENHANCER ../../../data/PRO-BAM/K562_PROseq_SRP045616.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/A549_hg38_StringentEnhancer_1000bp.bed.gz
sampleID=A549_Stringent.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/A549*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Stringent $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=A549_PROseq_SRX10323617-Stringent $ENHANCER ../../../data/PRO-BAM/A549_PROseq_SRX10323617.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/A549_hg38_LenientEnhancer_1000bp.bed.gz
sampleID=A549_Lenient.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/A549*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Lenient $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=A549_PROseq_SRX10323617-Lenient $ENHANCER ../../../data/PRO-BAM/A549_PROseq_SRX10323617.bam" >> $sampleID

ENHANCER=../../../data/Random_Coord/hg38_25KRand_1000bp.bed.gz
sampleID=A549_Random.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/A549*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Random $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=A549_PROseq_SRX10323617-Random $ENHANCER ../../../data/PRO-BAM/A549_PROseq_SRX10323617.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/HepG2_hg38_StringentEnhancer_1000bp.bed.gz
sampleID=HepG2_Stringent.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/HepG2*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Stringent $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=HepG2_GROseq_SRR5109940-Stringent $ENHANCER ../../../data/PRO-BAM/HepG2_GROseq_SRR5109940.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/HepG2_hg38_LenientEnhancer_1000bp.bed.gz
sampleID=HepG2_Lenient.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/HepG2*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Lenient $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=HepG2_GROseq_SRR5109940-Lenient $ENHANCER ../../../data/PRO-BAM/HepG2_GROseq_SRR5109940.bam" >> $sampleID

ENHANCER=../../../data/Random_Coord/hg38_25KRand_1000bp.bed.gz
sampleID=HepG2_Random.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/HepG2*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Random $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=HepG2_GROseq_SRR5109940-Random $ENHANCER ../../../data/PRO-BAM/HepG2_GROseq_SRR5109940.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/MCF7_hg38_StringentEnhancer_1000bp.bed.gz
sampleID=MCF7_Stringent.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/MCF7*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Stringent $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=MCF7_PROseq_SRP096173-Stringent $ENHANCER ../../../data/PRO-BAM/MCF7_PROseq_SRP096173.bam" >> $sampleID

ENHANCER=../../../data/Enhancer_Coord/MCF7_hg38_LenientEnhancer_1000bp.bed.gz
sampleID=MCF7_Lenient.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/MCF7*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Lenient $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=MCF7_PROseq_SRP096173-Lenient $ENHANCER ../../../data/PRO-BAM/MCF7_PROseq_SRP096173.bam" >> $sampleID

ENHANCER=../../../data/Random_Coord/hg38_25KRand_1000bp.bed.gz
sampleID=MCF7_Random.slurm
rm -f $sampleID
echo "$JOBSTATS" >> $sampleID
for file in ../../../data/BAM/MCF7*bam; do
        # Extract the base filename using basename
        filename=$(echo "$file" | sed 's|.*/\(.*\)\.bam|\1|')
        echo "$filename"
        echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=$filename\-Random $ENHANCER $file" >> $sampleID
done
echo "java -jar $SCRIPTMANAGER read-analysis tag-pileup --cpu=8 --gzip -t --combined --output-matrix=MCF7_PROseq_SRP096173-Random $ENHANCER ../../../data/PRO-BAM/MCF7_PROseq_SRP096173.bam" >> $sampleID

# Send slurm files out to cluster
for file in *.slurm; do
	echo $file
	sbatch $file;
done
