mkdir -p ../figures/fig2/panelb
mkdir -p ../figures/fig2/panelc

# Panel B
cd ../figures/fig2/panelb
grep "ROC" ../../../output-chr/results/*.txt > CHR_ROC.tab
# Panel C
cd ../panelc
grep "PRC" ../../../output-chr/results/*.txt > CHR_PRC.tab

