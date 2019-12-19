import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import surprise as spr

from scipy import sparse as sp
from itertools import groupby
from sklearn.preprocessing import PolynomialFeatures

# ========================= Data managing ========================= #

def load_data(path):
    """
    Load data from .csv file using Pandas library.
    @param path: a string, which is the path to the data file.
    @return: the data in Pandas format.
    """
    data = pd.read_csv(path)

    # split the Id column in user and movie indices
    data['User'] = data.Id.str.split('_').str[0].str[1:]
    data['Movie'] = data.Id.str.split('_').str[1].str[1:]
    data = data.rename(index=str, columns={"Prediction": "Rating"})
    data = data.drop(['Id'], axis=1)

    # shift the indices to start from 0
    data['User'] = data['User'].apply(lambda l: int(l) - 1)
    data['Movie'] = data['Movie'].apply(lambda l: int(l) - 1)

    return data

def explore_data(data, path):
    """
    Explore the data, print some statistics and show some plots.
    The computed statistics are the minimum, the maximum and the mean of number of ratings both given by users and given to the movies.
    The plots show the ratings given by users and the ratings given to movies.
    @param data: the data in Pandas format.
    @param path: the path to the folder where the plots are going to be saved, ending with /.
    """
    # users statistics
    rates_per_user = data['User'].value_counts()
    print('Rates per user:')
    print('min: ' + str(np.min(rates_per_user)))
    print('max: ' + str(np.max(rates_per_user)))
    print('mean: ' + str(np.mean(rates_per_user)))
    print('')

    # movie statistics
    rates_per_movie = data['Movie'].value_counts()
    print('Rates per movie:')
    print('min: ' + str(np.min(rates_per_movie)))
    print('max: ' + str(np.max(rates_per_movie)))
    print('mean: ' + str(np.mean(rates_per_movie)))

    # ratings per users
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(rates_per_user.values)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Ratings")
    ax1.grid(True);

    # ratings per movies
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(rates_per_movie.values)
    ax2.set_xlabel("Movies")
    ax2.set_ylabel("Ratings")
    ax2.grid(True);

    plt.suptitle("Number of ratings")
    plt.show()

    fig.savefig(path + 'rates_per.png')

def separate_indices(data):
    """
    Rearrange the columns of the data into a list of 3 N-tuples: [(user indices), (movie indices), (ratings)], where N is the number of given ratings.
    @param data: the data as a list of N 3-tuples, where each tuple is in format (user index, movie index, rating).
    @return: the rearranged data as a list if 3 N-tuples.
    """
    return list(zip(*data))

def transform_surprise(data):
    """
    Transform the data into a Surprise format.
    @param data: the data in a Pandas format.
    @return: the data in a Surprise format.
    """
    reader = spr.Reader(rating_scale=(1, 5))
    data_surprise = spr.Dataset.load_from_df(data[['User', 'Movie', 'Rating']], reader)
    return data_surprise

def sparse_matrix(data):
    """
    Create a sparse matrix from data using scipy library.
    @param data: the data in a Surprise format.
    @return: the data as a sparse matrix.
    """
    users, movies, ratings = separate_indices(data.build_testset())
    nb_users = max(users)+1
    nb_movies = max(movies)+1
    matrix = sp.lil_matrix((nb_users, nb_movies))
    matrix[users, movies] = ratings
    return matrix

def build_index_groups(train):
    """
    Group the indices by row or column index.
    @param train: the trainig set as a sparse matrix.
    @return: the indices of existing ratings, then same - grouped by row and grouped by column
    """
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def group_by(data, index):
    """
    Group a list of lists by given index.
    @param data: the data as a list of tuples (user, movie).
    @param index: the index to group by.
    @return: grouped data.
    """
    sorted_data = sorted(data, key=lambda x: x[index])
    grouped_data = groupby(sorted_data, lambda x: x[index])
    return grouped_data

def create_submission(prediction, data_path, submission_path):
    """
    Create a .csv file ready for the submission to AIcrowd platfowm. The prediction is rounded to values {1, 2, 3, 4, 5}.
    @param prediction: the prediction as a numpy array.
    @param data_path: the path to the test set data.
    @param submission_path: the path to the future submission file.
    """

    def round(x):
        if (x < 1):
            return 1
        elif (x > 5):
            return 5
        else:
            return x

    data = pd.read_csv(data_path)

    data['Prediction'] = prediction
    data['Prediction'] = data['Prediction'].apply(lambda l: np.rint(round(l)))
    data.to_csv(submission_path, sep=",")

# ========================= Learning ========================= #

def compute_rmse(test, predictions):
    """
    Compute the RMSE between the test set and predictions.
    @param test: the test set as a list of tuples.
    @param predictions: the predictions of ratings, not necessarily rounded.
    @return: RMSE.
    """
    test_rates = separate_indices(test)[2]
    rmse = np.linalg.norm(test_rates - predictions)/np.sqrt(len(test_rates))
    return rmse

def compute_error(data, user_features, movie_features, nonzero):
    """
    Compute the RMSE of the prediction on the training set, used in SGD and ALS.
    @param data: the test set.
    @param user_features: the user features matrix.
    @param movie_features: the movie features matrix.
    @param nonzero: indices where a rating exist.
    @return: RMSE.
    """
    mse = 0
    for row, col in nonzero:
        user_info = user_features[:, row]
        movie_info = movie_features[:, col]
        mse += (data[row, col] - user_info.T.dot(movie_info)) ** 2
    return np.sqrt(1.0 * mse / len(nonzero))

def polynomial_features(train, final_train, degree):
    """
    Compute the polynomial expansion of the data using the sklearn library. Used in model blending.
    @param train: the training set as a numpy array
    @param final_train: the true data set as a numpy array
    @param degree: the data is cross-multiplied up to the degree.
    @return: the training and the data sets expanded.
    """
    poly = PolynomialFeatures(degree)
    train_poly = poly.fit_transform(train)
    final_train_poly = poly.fit_transform(final_train)
    return train_poly, final_train_poly

