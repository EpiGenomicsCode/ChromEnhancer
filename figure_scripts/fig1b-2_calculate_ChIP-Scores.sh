module load anaconda

mkdir -p ../figures/fig1/panelb
cd ../figures/fig1/panelb

# Panel B
SCRIPTMANAGER=../../../bin/ScriptManager-v0.14.jar
JOBPATH=$PWD\/../../../figures/fig1/panelb

RENAME=../../../bin/rename_Matrix-Header.pl

INPUT=$PWD\/../../../figures/fig1/panelb
cd $INPUT
gunzip *.gz
for file in *.cdt; do
	java -jar $SCRIPTMANAGER read-analysis aggregate-data -l 3 --sum $file
	ID=$(echo "$file" | sed 's/\.cdt//g')
	perl $RENAME $ID\_SCORES.out temp
	mv temp $ID\_SCORES.out
done
gzip *.cdt

SCORE=../../../bin/average_Column.pl
for file in *.cdt.gz; do
        var=$(echo $file | awk -F"." '{print $1}')
        set -- $var
        ID="${1%.cdt.gz}"
        perl $SCORE $ID\_SCORES.out $ID\_AVG.out
done
cat *_AVG.out > ALL_SCORES.tab
rm *_SCORES.out *_AVG.out

AVG=../../../bin/calculate_AVG.py
python $AVG ALL_SCORES.tab temp

TRANSPOSE=../../../bin/restructure_Column_to_Matrix.py
python $TRANSPOSE temp temp2

HEATMAP=../../../bin/chart/generate_Matrix-Heatmap.py
python $HEATMAP temp2 Fig1B.svg

rm temp temp2
