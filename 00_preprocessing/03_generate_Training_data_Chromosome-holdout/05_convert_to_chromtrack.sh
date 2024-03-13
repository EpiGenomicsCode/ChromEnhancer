CONVERT=../../bin/convert_CDT-GZ_to_chromtrack.pl
NORMALIZE=../../bin/normalize_chromtrack.pl

FACTOR=1
CPU=0

HOLDOUT=../../data/CHR-HOLDOUT
cd $HOLDOUT

for file in *.cdt.gz; do
	tempID="${file/_StringentEnhancer/}"
	fileID="${tempID/.cdt.gz/}"
	# Convert CDT to chromtrack
	perl $CONVERT $file $FACTOR $fileID\_RAW.chromtrack &

	# Multi-thread to 8 cores
	let CPU++
	if [[ $CPU -eq 4 ]]; then
		wait
		CPU=0
	fi
done
wait

for file in *RAW.chromtrack; do
	fileID="${file/_RAW/}"
	# Normalize chromtrack by setting matrix max to 1 and scaling the rest
	perl $NORMALIZE $file $fileID &

	# Multi-thread to 8 cores
	let CPU++
	if [[ $CPU -eq 16 ]]; then
		wait
		CPU=0
	fi
done
wait

rm *RAW.chromtrack
gzip *.chromtrack

TRAIN=../../data/CHR-TRAIN
cd $TRAIN

for file in *.cdt.gz; do
	fileID="${file/.cdt.gz/}"
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

for file in *RAW.chromtrack; do
	fileID="${file/_RAW/}"
	# Normalize chromtrack by setting matrix max to 1 and scaling the rest
	perl $NORMALIZE $file $fileID &

	# Multi-thread to 8 cores
	let CPU++
	if [[ $CPU -eq 12 ]]; then
		wait
		CPU=0
	fi
done
wait

rm *RAW.chromtrack
gzip *.chromtrack
