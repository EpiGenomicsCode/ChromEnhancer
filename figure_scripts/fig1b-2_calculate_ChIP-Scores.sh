mkdir -p ../figures/fig1/panelb
cd ../figures/fig1/panelb

# Panel B
SCRIPTMANAGER=../../../bin/ScriptManager-v0.14.jar
JOBPATH=$PWD\/../../../figures/fig1/panelb

RENAME=/storage/home/wkl2/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/221130_EnhancerValidation/job/rename_Matrix-Header.pl

INPUT=/storage/home/wkl2/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/240308_Fig1-Gen/figures/fig1/panelb
cd $INPUT
gunzip *.gz
for file in *.cdt; do
	java -jar $SCRIPTMANAGER read-analysis aggregate-data -l 3 --sum $file
	ID=$(echo "$file" | sed 's/\.cdt//g')
	perl $RENAME $ID\_SCORES.out temp
	mv temp $ID\_SCORES.out
done
gzip *.cdt

SCORE=/storage/home/wkl2/bfp2/default/wkl2-WillLai/Enhancer-NN_Project/221130_EnhancerValidation/job/average_Column.pl

for file in *.cdt.gz; do
        var=$(echo $file | awk -F"." '{print $1}')
        set -- $var
        ID="${1%.cdt.gz}"
        perl $SCORE $ID\_SCORES.out $ID\_AVG.out
done
cat *_AVG.out > ALL_SCORES.tab

