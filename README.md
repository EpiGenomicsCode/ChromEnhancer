# Week 1: Docker Python and Statistics

## Prelab:

In order to prepare for this week's lecture, we first ask that the students view the following links and understand the material going into the lecture. 

### Docker information:

1. A simple 5 min introduction to [docker](https://youtu.be/_dfLOzuIg2o?t=14)
2. How to get set up using a [Dockerfile](https://youtu.be/LQjaJINkQXY)
3. Extra information about the Docker [process](https://youtu.be/LQjaJINkQXY)


### Python

This is a prereq for the class, however, just in case, there is a lot of documentation on using python. I specifically put this here to passive-aggressively stress the importance of python.

**In this course, I would strongly recommend also being familiar with the Anaconda environment and Jupyter notebooks**

### Statistics

All machine learning is statistics, please brush up on your calculus, linear algebra, and statistics knowledge.

## LAB

In this lab we will do the following:

1. Go over the dockerfile: 
2. start up a container and an image
3. check if there exists a GPU using Pytorch
4. Go over a few simple math and ML programs 
5. some other stuffs


## Homework:

1. Create your own Dockerfile that installs pandas, matplotlib, and other packages
2. Create a python script that reads the data found in titanic.csv
3. Process the data and make cool figures
4. save the data and have it propagate to your main computer.

# Resources

1. Download the container: docker build -t <tag_name> .
2. Run the container: docker run -it --rm --gpus all --name pytorch -v $PWD:/work <tag_name> 
3. start jupyter notebook: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root