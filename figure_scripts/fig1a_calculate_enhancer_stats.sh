mkdir -p ../figures/fig1/panela
cd ../figures/fig1/panela

# Panel A

# Cleanup and reset peak stat files
rm -rf ATAC_stats.tab STARR_stats.tab Enhancer_stats.tab
touch ATAC_stats.tab
touch STARR_stats.tab
touch Enhancer_stats.tab

# Calculate enhancer/peak statistics
ATAC=../../../data/ATAC_Cellline
for file in $ATAC/*.bed.gz; do
	echo $file >> ATAC_stats.tab
	gunzip -c $file | wc -l >> ATAC_stats.tab
done

STARR=../../../data/STARR_Cellline
for file in $STARR/*.bed.gz; do
	echo $file >> STARR_stats.tab 
	gunzip -c $file | wc -l >> STARR_stats.tab 
done

ENH=../../../data/Enhancer_Coord
for file in $ENH/*Enhancer.bed.gz; do
	echo $file >> Enhancer_stats.tab
	gunzip -c $file | wc -l >> Enhancer_stats.tab
done
