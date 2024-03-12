#chromHMM states
wget -O A549_BSS00007_18_CALLS_segments.bed.gz https://personal.broadinstitute.org/cboix/epimap/ChromHMM/observed_aux_18_hg38/CALLS/BSS00007_18_CALLS_segments.bed.gz
wget -O HepG2_BSS00558_18_CALLS_segments.bed.gz https://personal.broadinstitute.org/cboix/epimap/ChromHMM/observed_aux_18_hg38/CALLS/BSS00558_18_CALLS_segments.bed.gz
wget -O K562_BSS00762_18_CALLS_segments.bed.gz https://personal.broadinstitute.org/cboix/epimap/ChromHMM/observed_aux_18_hg38/CALLS/BSS00762_18_CALLS_segments.bed.gz
wget -O MCF7_BSS01226_18_CALLS_segments.bed.gz https://personal.broadinstitute.org/cboix/epimap/ChromHMM/observed_aux_18_hg38/CALLS/BSS01226_18_CALLS_segments.bed.gz

mkdir -p ../../data/chromHMM_hg38/
mv *.bed.gz ../../data/chromHMM_hg38/
