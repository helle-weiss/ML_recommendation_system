# Machine Learning — Project 2, Recommender systems

This is the second project of the EPFL Machine Learning course. The goal was to make recommendations to users based on their previous ratings. The prediction was then submitted to the competition arena on [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019/leaderboards) and shared the second place with some other teams, with **RMSE 1.017**. 
The detailed explanations and reasoning can be found in the [report](https://github.com/helle-weiss/ML_project2/blob/master/report_nessler.pdf). 

## External libraries

In order to run this project, you will need the following libraries:

- numpy
- scipy
- matplotlib
- pandas
- itertools
- sklearn
- surprise
- keras
- pickle

The command `pip install <name>` should work.

## How to run

The files `run.ipynb` and `run.py` are equivalent. Simply run any of them without changing anything to get the same results: this will use the trained models. If you want to train the models again yourself, change the boolean `load_models` to `False` in the beginning of the `run` file. The training will take some time.

## Data

In this project, 10000 users rate 1000 items, for example movies, from 1 to 5. The data consists of these ratings, given by users to movies, and the goal is to predict new ratings. 
The training set is a [1176952 x 2] matrix, where the first column consists of pairs of users and movies and the second column consists of ratings. The entries of the first column are in the following format: `rXcY`, where `X` and `Y` are integers; this means that a user `X` has rated an item `Y`. The item in the second column and the same row represents the corresponding rating.

## Files organization

The content of this project is composed of several folders and files:

   - `data/`:
      - `data_train.csv`: the dataset used for training, in the format specified above.
      - `data_test.csv`: the dataset with ratings to predict. For now, all the ratings are arbitrarily set to 3.
      
   - `figures/`: the folder to save the figures during the data exploration.
   
   - `final_models/`: the folder containing the trained models on the whole dataset.
   
   - `models/`: the folder containing the trained models on the training set.

   - `run.py`: the main file. By running it, we load the data and the models, combine them and output a predictions file to the competition arena. If `load_models` is set to `False`, we train the models instead.
   
   - `run.ipynb`: the Jupyter Notebook file, equivalent to the `run.py` file.
   
   - `models.py`: the file containing all the models: Baseline, KNN Baseline, Basic KNN, KNN-means, KNN-zscore, SVD, SVD++, NMF, Slope One, SGD, ALS, Ridge regression and Neural network.
   
   - `helpers.py`: the file containing all helper functions: load data, explore data, transform data from one format to another, create a submission file, compute the error, and do a polynomial feature expansion.
   
   - `grid_and_CV.py`: the file containing the grid search functions. The cross-validation has been done by changing the seed when splitting the data set.
