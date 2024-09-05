set -exo
module load anaconda3_cpu
#source activate /scratch/bbse/wklai/EnhancerNN/bedtools

#MODEL=model5

IDIR=../output-cell/xai
ODIR=../figures/fig4/panela
[ -d $ODIR ] || mkdir -p $ODIR

SDIR=../figures/sfig3
[ -d $SDIR ] || mkdir -p $SDIR

# Heatmap
HEATMAP=bin/visualize_XAI-hierarchical.py

REP=2
FILEHEAD="CLD_study_-_test_-_valid_-"
FILETAIL="clkeep_K562-HepG2-A549_chkeep_CTCF-H3K4me3-H3K27ac-p300-PolII-H3K36me3-H3K27me3-H3K4me1_type-${REP}_epoch_20.pt"

MODEL=1
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

MODEL=2
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

MODEL=3
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

MODEL=4
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

MODEL=5
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

MODEL=6
FILEID="${FILEHEAD}_model${MODEL}_$FILETAIL"
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_saliency.tsv --linkVMax 50 --outputFile $ODIR/model$MODEL\_saliency.svg --outputDendrogram --outputDendrogramFile $ODIR/model$MODEL\_dendrogram.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_gradient.tsv --linkVMin 0 --outputFile $ODIR/model$MODEL\_gradient.svg
python $HEATMAP --clustData $IDIR/$FILEID\_xai_orig.tsv --linkData $IDIR/$FILEID\_xai_shap.tsv --linkVMin 0 --linkVMax 1 --outputFile $SDIR/model$MODEL\_shap.svg #--outputDendrogram --outputDendrogramFile $SDIR/model$MODEL\_dendrogram.svg

ODIRB=../figures/fig4/panelb
[ -d $ODIRB ] || mkdir -p $ODIRB
mv $ODIR/*saliency* $ODIRB
