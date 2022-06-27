FROM  pytorch/pytorch

RUN conda install pyg -c pyg
RUN conda install anaconda 
RUN apt-get update
RUN apt-get install libxrender1 -y
RUN apt-get install -y libsm6 libxext6 -y
RUN apt-get install -y libxrender-dev -y
RUN pip install pysmiles m2p rdkit-pypi opencv-python torchviz
RUN conda install -c conda-forge tensorboard
RUN apt-get install graphviz -y
RUN conda install -c conda-forge deepchem
