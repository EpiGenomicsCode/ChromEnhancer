# Chromatin Enhancer Study

Multi-cellular organisms exhibit diverse cellular phenotypes, largely due to the complex interplay of regulatory complexes distributed throughout the eukaryotic genome, which enhance specific gene programs across distinct networks of cell types. Enhancer prediction using deep-learning methods typically incorporates DNA sequence and epigenomic datasets to improve accuracy and precision, but this approach is limited by the clonal expansion of DNA mutations in cancers. To address this, we explored the feasibility of enhancer prediction using only epigenomic chromatin datasets and successfully developed a cell-type invariant enhancer prediction platform that utilized only chromatin marks. This approach serves as a proof-of-concept for future biomedical applications, particularly with the potential for reference-genome free alignment of epigenomic datasets.

# Steps

## 1. Preprocessing

* Navigate to the Preprocessing folder and follow the instructions

## 2. Building the environment (From .devconatiner)

* Install Docker Desktop
* Install VSCode
* Install Python extension with VSCode
* Install DevContainer extension with VSCode
* "Shift - CMD - P" > Open Folder in Container

## 3. Running different studies

There are currently three different studies that can be performed:

1. Sequence: This study ...
2. Parameter: This study ...
3. Cell Line Dropout: This study ...

### FAQ

1. How do I add my own model?
* Navigate to the util/Model folder and add your model as a new python file
* Edit the loadModel function in the util.py file
