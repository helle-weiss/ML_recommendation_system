import numpy as np
import scipy as sp
import surprise as spr

from helpers import *

# for the blending:
from sklearn.linear_model import Ridge 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# ========================= Surprise models ========================= #

def get_predictions(predictions):
    """
    Transform the Surprise format to a numpy array. Also print RMSE.
    @param predictions: predictions returned by a model from the Surprise library.
    @return: predictions in a numpy array.
    """
    spr.accuracy.rmse(predictions)
    result = np.zeros(len(predictions))
    for i, pred in enumerate(predictions):
        result[i] = pred.est
    return result

def baseline(train, test):
    """
    Run Baseline model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    algo = spr.BaselineOnly()
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def baselineKNN(train, test, user_bool):
    """
    Run KNN Baseline model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @param user_bool: if True, runs the user based KNN baseline. Otherwise, runs item based KNN baseline.
    @return: the predictions in a numpy array.
    """
    algo = spr.KNNBaseline(name='pearson_baseline', user_based=user_bool)
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def baselineKNN_item(train, test):
    """
    Run the item-based KNN Baseline model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    return baselineKNN(train, test, False)

def baselineKNN_user(train, test):
    """
    Run the user-based KNN Baseline model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    return baselineKNN(train, test, True)

def basicKNN(train, test):
    """
    Run the basic KNN model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    algo = spr.KNNBasic()
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def meansKNN(train, test):
    """
    Run the KNN means model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    algo = spr.KNNWithMeans()
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def zscoreKNN(train, test, user_bool):
    """
    Run the KNN zscore model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @param user_bool: if True, runs the user based KNN zscore. Otherwise, runs item based KNN zscore.
    @return: the predictions in a numpy array.
    """
    algo = spr.KNNWithZScore(user_based=user_bool)
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def zscoreKNN_user(train, test):
    """
    Run the user-based KNN zscore model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    return zscoreKNN(train, test, True)

def zscoreKNN_item(train, test):
    """
    Run the item-based KNN zscore model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    return zscoreKNN(train, test, False)

def SVD(train, test, rate):
    """
    Run the SVD model from Surprise library. The number of factors is 40. The number of iterations is 20.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @param rate: the learning rate of all parameters.
    @return: the predictions in a numpy array.
    """
    algo = spr.SVD(n_factors=40, lr_all=rate)
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def SVDpp(train, test, rate):
    """
    Run the SVD++ model from Surprise library. The number of factors is 40. The number of iterations is 20.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @param rate: the learning rate of all parameters.
    @return: the predictions in a numpy array.
    """
    algo = spr.SVDpp(n_factors=40, lr_all=rate, verbose=True)
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def NMF(train, test):
    """
    Run the NMF model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    algo = spr.NMF()
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

def slopeOne(train, test):
    """
    Run the Slope One model from Surprise library.
    @param train: the training set in the Surprise format.
    @param test: the test set in the Surprise format.
    @return: the predictions in a numpy array.
    """
    algo = spr.SlopeOne()
    algo.fit(train)
    predictions = algo.test(test)
    return get_predictions(predictions)

# ========================= SGD ========================= #

def init_MF(train, num_features):
    """
    Initialize the feature vectors for matrix factorization.
    @param train: the training set as a sparse matrix.
    @param num_features: the number of features for the feature vectors.
    @return: a matrix of user features and a matrix of movie features.
    """
    nb_users, nb_movies = train.shape

    user_features = np.random.rand(num_features, nb_users)
    movie_features = np.random.rand(num_features, nb_movies)

    # start by movie features
    movie_nnz = train.getnnz(axis=0)
    movie_sum = train.sum(axis=0)

    for ind in range(nb_movies):
        movie_features[0, ind] = movie_sum[0, ind] / movie_nnz[ind]

    return user_features, movie_features

def SGD(train, test, num_features, lambda_user, lambda_movie, gamma, stop_criterion):
    """
    Run the SGD model.
    @param train: the training set as a sparse matrix.
    @param test: the test set.
    @param num_features: the number of features for the initialization matrices.
    @param lambda_user: the lambda parameter for user features.
    @param lambda_movie: the lambda parameter for movie features.
    @param gamma: the gamma parameter to control the updating of the features.
    @param stop_criterion: the learning will stop when the difference between two consecutive errors is smaller than stop_criterion.
    @return: the prediction as a numpy array.
    """
    errors = [10, 9] # initialization, well be removed in the end
    iteration = 0
    np.random.seed(988)

    # initialize feature vectors
    user_features, movie_features = init_MF(train, num_features)
    # get the indices of non-zero entries of the data
    nz_train, _, _ = build_index_groups(train)

    # run SGD
    print("Starting the SGD algorithm...")
    while np.abs(errors[-2] - errors[-1]) > stop_criterion:
        iteration += 1
        np.random.shuffle(nz_train)
        gamma = gamma/2

        # update user and movie features
        for user, movie in nz_train:
            distance = train[user, movie] - user_features[:, user].T.dot(movie_features[:, movie])
            user_features[:, user] += gamma * (distance * movie_features[:, movie] - lambda_user * user_features[:, user])  
            movie_features[:, movie] += gamma * (distance * user_features[:, user] - lambda_movie * movie_features[:, movie])

        error = compute_error(train, user_features, movie_features, nz_train)
        print("Iteration {}, current error on the training set: {}.".format(iteration, error))
        errors.append(error)

    # remove the initialization
    errors.remove(10)
    errors.remove(9)
    
    prediction = user_features.T.dot(movie_features)
    
    return prediction[test[0], test[1]]

# ========================= ALS ========================= #

def update_user_feature(train, movie_features, lambda_user, ratings_per_user, nz_user_movie_indices):
    """
Update the user features matrix for the ALS algorithm.
    @param train: the training set as a sparse matrix.
    @param movie_features: the movie features matrix.
    @param lambda_user: the lambda parameter for user features.
    @param ratings_per_user: the number of movies the user has rated.
    @param nz_user_movie_indices: the indices of the movies the user has rated.
    @return: the updated feature matrix.
    """
    nb_users = ratings_per_user.shape[0]
    num_features = movie_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_features)
    updated_user_features = np.zeros((num_features, nb_users))

    for user, movies in nz_user_movie_indices:
        # extract the columns corresponding to the prediction for given movie
        M = movie_features[:, movies]
        # update column row of user features
        V = M @ train[user, movies].T
        A = M @ M.T + ratings_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_movie_feature(train, user_features, lambda_movie, ratings_per_movie, nz_movie_user_indices):
    """
Update the user features matrix for the ALS algorithm.
    @param train: the training set as a sparse matrix.
    @param user_features: the user features matrix.
    @param lambda_movie: the lambda parameter for movie features.
    @param ratings_per_movie: the number of users who rated the movie.
    @param nz_movie_user_indices: the indices of the users who rated the movie.
    @return: the updated feature matrix.
    """
    nb_movies = ratings_per_movie.shape[0]
    num_features = user_features.shape[0]
    lambda_I = lambda_movie * sp.eye(num_features)
    updated_movie_features = np.zeros((num_features, nb_movies))

    for movie, users in nz_movie_user_indices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[users, movie]
        A = M @ M.T + ratings_per_movie[movie] * lambda_I
        X = np.linalg.solve(A, V)
        updated_movie_features[:, movie] = np.copy(X.T)
    return updated_movie_features

def ALS(train, test, num_features, lambda_user, lambda_movie, stop_criterion):
    """
    Run the ALS model.
    @param train: the training set as a sparse matrix.
    @param test: the test set.
    @param num_features: the number of features for the initialization matrices.
    @param lambda_user: the lambda parameter for user features.
    @param lambda_movie: the lambda parameter for movie features.
    @param stop_criterion: the learning will stop when the difference between two consecutive errors is smaller than stop_criterion.
    @return: the prediction as a numpy array.
    """
    errors = [10, 9] # initialization, well be removed in the end
    iteration = 0
    np.random.seed(988)

    # initialize feature vectors
    user_features, movie_features = init_MF(train, num_features)
    # get the indices of non-zero entries of the data
    ratings_per_user, ratings_per_movie = train.getnnz(axis=1), train.getnnz(axis=0)
    # group the indices by row or column index
    nz_train, nz_user_movie_indices, nz_movie_user_indices = build_index_groups(train)

    # run ALS
    print("Starting the ALS algorithm...")
    while np.abs(errors[-2] - errors[-1]) > stop_criterion:
        iteration += 1
        # update user and movie features
        user_features = update_user_feature(
            train, movie_features, lambda_user,
            ratings_per_user, nz_user_movie_indices)
        movie_features = update_movie_feature(
            train, user_features, lambda_movie,
            ratings_per_movie, nz_movie_user_indices)

        error = compute_error(train, user_features, movie_features, nz_train)
        print("Iteration {}, current error on the training set: {}.".format(iteration, error))
        errors.append(error)
    
    errors.remove(10)
    errors.remove(9)
    
    prediction = user_features.T.dot(movie_features)
    
    return prediction[test[0], test[1]]

# ========================= Model blending ========================= #

def ridge_regression(train, test, data, seed):
    """
    Run the Ridge Regression from sklearn library. Alpha is 0.01.
    Compute the weights of each model on training set and then make a prediction using weighted models on data set.
    @param train: the matrix with the predictions of different models on a training set, one prediction per column.
    @param test: the corresponding test predictions.
    @param data: the matrix with the predictions of different models on a true data set, one prediction per column.
    @param seed: the fixed seed.
    @return: the prediction as a numpy array.
    """
    reg = Ridge(alpha=0.01, fit_intercept=False, random_state=seed)
    reg.fit(train, test)

    final_result = 0
    for i in range(data.shape[1]):
        final_result = final_result + reg.coef_[i] * data[:, i]
    return final_result

def neural_net(train, test, data, epochs, patience):
    """
    Run a neural net from Keras library.
    @param train: the matrix with the predictions of different models on a training set, one prediction per column.
    @param test: the corresponding test predictions.
    @param data: the matrix with the predictions of different models on a true data set, one prediction per column.
    @param epochs: numbers of epochs to run.
    @param patience: the learning will stop earlier if there has been no improvement in the past #patience epochs.
    @return: the prediction as a numpy array.
    """
    model = Sequential()
    model.add(Dense(12, input_dim=train.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    es = EarlyStopping(monitor="mean_squared_error", mode="min", verbose=1, patience=patience)
    model.fit(train, test, epochs=epochs, batch_size=10, callbacks=[es])
    prediction = model.predict(data) # an array where each element is another array with one element inside
    return prediction.flatten()