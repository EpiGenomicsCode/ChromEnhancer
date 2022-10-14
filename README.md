Chromatin Enhancer prediction

Descriptive words for the project

# How to run a study

## Preprocessing
* GOTO Preprocessing Folder

## Initializing the environment
* Download the repo and navigate to the main file
* Edit the startDocker.sh command as needed for GPU support
* Run the command sh startDocker.sh
* Navigate to the /work directory in docker


## Running the model
* In the runmodels.py file, you can edit the training, testing, and validation data alongside the different models you want to run and the number of epochs.
* Run the runmodels.py file in the docker environment
* All model weights and biases are saved in output/model_weight_bias

## Visualizing the data
* To create the visualizations, you can run the visualizeData.py file
* This will give you the PRC and ROC curves; individual images in output/prc, and output/roc, grouped images are in dataVis
* This will save the values in output/coord

## Doing a Swarm Study on the input
* To do a swarm Study you can run the swarmStudy.py file
* The output of the last position and its score is given in output/swarm while the visualization of clustering is given in output/cluster

# FAQ

## Adding models
* To add your  model first create a new model class in the Chrom_Proj/model.py file
* Navigate to the Chom_Proj/runner.py file in the runner method and assign a new number to your class(i.e., lines 168-178) 
* In the runModels.py file add your number mapping to the model's list
* Congrats you have added a new model!
