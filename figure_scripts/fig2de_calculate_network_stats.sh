mkdir -p ../figures/fig2/paneld
mkdir -p ../figures/fig2/panele

# Panel D
cd ../figures/fig2/paneld
grep "ROC" ../../../output-cell/results/*.txt > CHR_ROC.tab
# Panel E
cd ../panele
grep "PRC" ../../../output-cell/results/*.txt > CHR_PRC.tab

