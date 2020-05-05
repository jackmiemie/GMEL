# Intro

This repo contains the code and data for our paper ***Learning Geo-Contextual Embeddings for Commuting Flow Prediction***, published at the Thirty-Fourth AAAI Conference on Artificial Intelligence (2020).  Click [here](https://arxiv.org/abs/2005.01690) for the full paper.

GMEL makes use of land use information, such as NYC's [PLUTO](https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page), and commuting trips information to study the problem of commuting flow prediction using graph neural network.

![GMEL Framework](/img/framework.png)


# Table of contents

- [Intro](#intro)
- [Table of contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Structure](#structure)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

# Prerequisites

The code is written using Python. The following Python packages are required:

```
Python 3.x
Pytorch 1.4.0
DeepGraphLibrary 0.4.1
Scikit-Learn 0.21.3
Numpy 1.17.2
Pandas 0.25.1
numpy_indexed
```


Other tools:

```
Tensorboard
```

# Structure

`code` directory contains all the code to run the experiments:

+ `01train_GMEL.py` is the code to run GMEL training with multiple experimental settings. It will import `train.py` to train with each setting.

+ `02train_Predictor.py` is the code to run predictor training corresponding to the settings of GMEL. It will need the embeddings generated from `01train_GMEL.py`.

+ `train.py` is the code to train a single GMEL with a specific setting. If you are interested in the process of training a graph neural network, this is the code you should read.

+ `model.py` is the code for GMEL. Basically, it is a graph neural network combined with the interface of multitask loss, generating embeddings, etc. If you are interested in the multitask learning and graph neural network, this is the code you should read.

+ `layers.py` is the code for graph neural network layers. If you are interested in the message propagation process in graph neural network, this is the code you should read.

+ `utils.py` is the code for our-own-written tools, e.g. data loader, mini-batch generator, evaluation metrics etc. If you are interested in how we preprocess the data, this is the code you should read.

`data` directory contains all the data described in our paper.

+ `LODES` contains the commuting trips data collected from LODES. We have randomly split the data into three pieces, i.e. train, validation and test, with the ratio of 6 : 2 : 2. You could merge these dataset and shuffle to create your own dataset if you like.

+ `PLUTO` contains the aggregated census tract features from PLUTO. Notice that the presented data are preprocessed using location quotient. Location quotient is a relative measure which tells how salient is the feature of a sample in contrast to the entire sample set.

+ `CensusTract2010` contains the census tract adjacency matrix and a node ID mapping table. The census tract version we used is 2010. The original census tract ID is preserved so that you can check the location of census tract on the Internet.

+ `OSRM` contains the distance matrix of census tracts measured by OSRM.

# Usage

Step 1. Set your working director to `code`

Step 2. Run `python 01train_GMEL.py`. 
This might take a long time if you add more experimental settings. While training, you can check the running status in `code/log` or open tensorboard setting logdir to `code/runs` to check the gradient descent process, etc.

Step 3. Having finished Step 2, run `python 02train_Predictor.py`. 
When finished running the code, you can check the `code/outputs` directory for the testing performance of each GMEL setting. If you want to explore the model, `code/models` stores all the models. If you are interested in the generated embeddings, you can check `code/embeddings` to get the embeddings using Numpy.

# Citation

```
@inproceedings{liu2020gmel,
  title={Learning Geo-Contextual Embeddings for Commuting Flow Prediction},
  author={Liu, Zhicheng and Miranda, Fabio and Xiong, Weiting and Yang, Junyan and Wang, Qiao and Silva, Claudio T.},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

# Contact

Please send any questions you might have about the code and/or the algorithm to zhi-cheng-liu AT seu.edu.cn (remove dash `-` in the address).

