
## Dependencies

```
pip install venny4py

```
```
conda create -n validation -c bioconda -c conda-forge numpy pandas dask scipy fastcluster matplotlib
```


# Execute scripts
```
bash 1_CallEnhancers.sh
bash 2_VennSplit.sh
bash 3_GenomeWideCorrelation.sh
bash 4_ENCODE-Peak-Enrichment.sh
```

See header of each shell script for more details

## ENCODE_Peak_Metadata.txt

```
# Establish search URL for retrieving samples
URL="https://www.encodeproject.org/search/?type=File&file_format_type=narrowPeak&file_type=bed+narrowPeak&assay_title=TF+ChIP-seq&biosample_ontology.term_name=K562&biosample_ontology.term_name=A549&biosample_ontology.term_name=MCF-7&biosample_ontology.term_name=HepG2&assembly=GRCh38&status=released&format=json&limit=all"

# Download metadata
python bin/get_ENCFF-Peak_from_ENCODEsearch.py --preferred -i $URL -o metadata.tmp

# Pull unique CL-TARGET combinations (preserve header)
head -1 metadata.tmp > ENCODE_Peak_Metadata.txt
sed '1d' metadata.tmp | sort -uk3,4 >> ENCODE_Peak_Metadata.txt

# Clean-up
rm metadata.tmp
```
