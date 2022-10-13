Chromatin Enhancer prediction

Words

# How to run study

## Preprocessing
* I am not sure

## Initializing the environment
* Download the repo and navigate to the main file
* Edit the startDocker.sh command as needed for GPU support
* run the command sh startDocker.sh
* navigate to the /work directory in docker


## Running the model
* In the runmodels.py file, you can edit the training, testing, and validation data alongside the different models you want to run and the number of epochs.
* run the runmodels.py file in the docker environment

## visualizing the data
* In order to create the visualizations, you can run the visualizeData.py file

## Doing a Swarm Study on the input
* To do a swarm Study you can run the swarmStudy.py file

# FAQ

# Adding models
* To add your own model first create a new model class in the Chrom_Proj/model.py file
* Navigate to the Chom_Proj/runner.py file in the runner method and assign a new number to your class(i.e., lines 168-178) 
* In the runModels.py file add your number mapping to the model's list
* Congrats you have added a new model!
