from models import *
import numpy as np

def grid_search_SVD(train, test, epochs, rates, bool_SVD):
    """
    Find the best parameters for SVD and SVD++ models.
    @param train: training set in Surprise format.
    @param test: test set in Surprise format.
    @param epochs: a list of epochs to try.
    @param rates: a list of rates to try.
    @param bool_SVD: if True, tests SVD. Otherwise tests SVD++.
    @return: the best epoch and the best rate.
    """
    rmse_SVD = []
    params_SVD = []
    iteration = 0
    for e in epochs:
        for r in rates:
            iteration += 1
            print("Iteration " + str(iteration))
            if bool_SVD:
                # pred = SVD(train, test, e, r) -- the epoch argument has been removed from SVD() as it is found
                pred = SVD(train, test, r)
            else:
                # pred = SVDpp(train, test, e, r) -- the epoch argument has been removed from SVD() as it is found
                pred = SVDpp(train, test, r)
            rmse_SVD.append(compute_rmse(test, pred))
            params_SVD.append([e, r])
    # best RMSE score and parameters
    bestRMSE = min(rmse_SVD)
    index = rmse_SVD.index(bestRMSE)
    bestP = params_SVD[index]
    print("Best RMSE {} is obtained with parameters: epoch = {}, rate = {}.".format(bestRMSE, bestP[0], bestP[1]))

    return bestP[0], bestP[1]

def grid_search_MF(train, test, num_features, lambda_users, lambda_movies, gammas, bool_ALS):
    """
    Find the best parameters for ALS and SGD models.
    @param train: training set as a sparse matrix.
    @param test: test set as a sparse matrix.
    @param num_features: a list of num_features to try.
    @param lambda_users: a list of lambda_users to try.
    @param lambda_movies: a list of lambda_movies to try.
    @param gammas: a list of gammas to try. Leave [] if ALS is tested.
    @param bool_ALS: if True, tests ALS. Otherwise tests SGD.
    @return: the best parameters.
    """
    rmse_MF = []
    params_MF = []
    iteration = 0
    for f in num_features:
        for lu in lambda_users:
            for lm in lambda_movies:
                if bool_ALS:
                    iteration += 1
                    print("Iteration " + str(iteration))
                    pred = ALS(train, test, f, lu, lm, 1e-4)
                    rmse_MF.append(np.linalg.norm(test[2] - pred)/np.sqrt(len(test[2])))
                    params_MF.append([f, lu, lm])

                else:
                    for g in gammas:
                        iteration += 1
                        print("Iteration " + str(iteration))
                        pred = SGD(train, test, f, lu, lm, g, 1e-4)
                        rmse_MF.append(np.linalg.norm(test[2] - pred)/np.sqrt(len(test[2])))
                        params_MF.append([f, lu, lm, g])

    # best RMSE score and parameters
    bestRMSE = min(rmse_MF)
    index = rmse_MF.index(bestRMSE)
    bestP = params_MF[index]
    print("Best RMSE {} is obtained with parameters: {}.".format(bestRMSE, bestP))

    return bestP

# The cross-validation is easily done by changing the seed when separating the data set into a training and a test set..