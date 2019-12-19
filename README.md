# Machine Learning — Project 2, Recommender systems

This is the second project of the EPFL Machine Learning course. The goal was to make recommendations to users based on their previous ratings. The prediction was then submitted to the competition arena on [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019/leaderboards) and shared the first place with some other teams, with RMSE 1.017. 

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

## How to run

The files `run.ipynb` and `run.py` are equivalent. Simply run any of them without changing anything to get the same results: this will use the trained models. If you want to train the models again yourself, change the boolean `load_models` to `False` in the beginning of the `run` file. The training will take some time.

## Files organization

The content of this project is composed of several folders and files:

#### Data

   - `data/`:
      - `data_train.csv`: the dataset used for training. It is not made public, but is necessary to run the project.
      - `data_test.csv`: the dataset used to test the model. It is not made public, but is necessary to run the project.

   - `run.py`: the main file. By running it, we load the data, train the model, test it and output a predictions file to the competition arena.

   - `implementations.py`: the file containing the six basic models for machine learning: 
      - `least_squares_GD`: linear regression using gradient descent.
      - `least_squares_SGD`: linear regression using stochastic gradient descent.
      - `least_squares`: least squares regression using normal equations.
      - `ridge_regression`: ridge regression using normal equations.
      - `logistic_regression`: logistic regression using gradient descent.
      - `reg_logistic_regression`: regularized logistic regression using gradient descent.

   - `utils.py`: the file containing all other functions useful for the project:
      - `remove999`: clean the data. 
      - `standardize`: standardize the data.
      - `build_poly`: build a feature extension.
      - `accuracy`: compute the accuracy.
      - `compute_loss`: compute the loss.
      - `compute_gradient`: compute the gradient.
      - `sigmoid`: the sigmoid function.


   - `proj1_helpers.py`: the file with given functions: to load the data and make the predictions.

   - `given_helpers.py`: the file with some functions given during the labs before the project: for example, batch_iter.

   - `validation.py`: the file used to do a cross validation.
      - `split_data`: splits the data according to a ratio.
      - `run_model`: runs the given model.
      - `build_k_indices`: shuffles the indices of the data.
      - `cross_validation`: runs the cross validation.
