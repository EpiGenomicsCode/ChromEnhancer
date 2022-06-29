
# TODO:

1. Figure out Data
    * ~~Meeting 6/27/22: Create a dataloader for the chromatin data with the label, similar to pandas dataframe maybe~~
    * Remove duplicates from data
    * Fix make embedd alignment
    * Look into tfrecords and training time 
    * Maybe data augmentation
    * Add Seq data to the dataset **(is this needed)**
    * Create a validation dataset
2. Build Network
    * ~~Simple Neural Network~~
    * A Transformer of some kind?
3. ~~Run Network~~
    * ~~Not Yet~~
4. Collect Results
    * AUC and PRC


# Changelog
## 6/28/22
---
### Done
    1. Created a Dataset that contains the label and the chromtracks
    2. Created a default model for binary classification
    3. Created the training and testing loop

## 6/29/22 (Olivia)
    1. Initialize preprocessing scripts directory structure
    2. Add preprocessing scripts for downloading ENCODE data
    3. Add preprocessing scripts for calling Peak Enhancers (partial)
        - waiting on covariate files from gersteinlab
        - testing runs without covariates until then
    4. Initialize README
        - describe all above
        - sketch out formatting step in README

# Resources

* Something like: https://github.com/yifengtao/genome-transformer
* Or maybe: https://towardsdatascience.com/bringing-bert-to-the-field-how-to-predict-gene-expression-from-corn-dna-9287af91fcf8

# Useful Commands
1. Download the container: docker build -t <tag_name> .
2. Run the container: docker run -it --rm --gpus all --name pytorch -v $PWD:/work <tag_name> 
3. start jupyter notebook: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
