CONVERT=../../00_preprocessing/bin/convert_CDT-GZ_to_chromtrack.pl
NORMALIZE=../../00_preprocessing/bin/normalize_chromtrack.pl

FACTOR=1
CPU=0

HOLDOUT=../../data/CELL-HOLDOUT
cd $HOLDOUT

# for file in *.cdt.gz; do
# 	fileID="${file/.cdt.gz/}"
# 	# Convert CDT to chromtrack
# 	perl $CONVERT $file $FACTOR $fileID\_RAW.chromtrack &
# 
# 	# Multi-thread to 8 cores
# 	let CPU++
# 	if [[ $CPU -eq 12 ]]; then
# 		wait
# 		CPU=0
# 	fi
# done
# wait
# 
# for file in *_0_*RAW.chromtrack; do
# 	fileID1="${file/_0_/_1_}"
# 	fileID2="${file/_0_/_2_}"
# 	fileID3="${file/_0_/_3_}"
# 	fileID4="${file/_0_/_4_}"
# 	fileID5="${file/_0_/_5_}"
# 	fileID="${file/_0_/_}"
# 	echo $fileID
# 	cat $file $fileID1 $fileID2 $fileID3 $fileID4 $fileID5 > $fileID
# 	rm $file $fileID1 $fileID2 $fileID3 $fileID4 $fileID5
# done
# 
# for file in *RAW.chromtrack; do
# 	fileID="${file/_RAW/}"
# 	# Normalize chromtrack by setting matrix max to 1 and scaling the rest
# 	perl $NORMALIZE $file $fileID &
# 
# 	# Multi-thread to 8 cores
# 	let CPU++
# 	if [[ $CPU -eq 12 ]]; then
# 		wait
# 		CPU=0
# 	fi
# done
# wait
# 
# rm -f *RAW.chromtrack
# gzip *.chromtrack

TRAIN=../../data/CELL-TRAIN
cd $TRAIN

for file in *.cdt.gz; do
	fileID="${file/.cdt.gz/}"
	echo $fileID
	# Convert CDT to chromtrack
	perl $CONVERT $file $FACTOR $fileID\_RAW.chromtrack &

	# Multi-thread to 8 cores
	let CPU++
	if [[ $CPU -eq 12 ]]; then
		wait
		CPU=0
	fi
done
wait

for file in *_0_*RAW.chromtrack; do
      fileID1="${file/_0_/_1_}"
      fileID="${file/_0_/_}"
      echo $fileID
      cat $file $fileID1 > $fileID
      rm $file $fileID1
done

for file in *RAW.chromtrack; do
	fileID="${file/_RAW/}"
	echo $fileID
	# Normalize chromtrack by setting matrix max to 1 and scaling the rest
	perl $NORMALIZE $file $fileID &
	# Multi-thread to 8 cores
	let CPU++
	if [[ $CPU -eq 8 ]]; then
		wait
		CPU=0
	fi
done
wait

rm -f *RAW.chromtrack
gzip *.chromtrack
