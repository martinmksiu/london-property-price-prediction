import pandas as pd
import numpy as np
import scipy.stats as st
from os.path import dirname, realpath
cwd = dirname(realpath(__file__))

from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

print("Loading data...\n")
# train = pd.read_csv(cwd+"/Data/train.csv")
# test = pd.read_csv(cwd+"/Data/test.csv")

train = pd.read_csv(cwd+"/DATA/train2.csv")
test = pd.read_csv(cwd+"/DATA/test2.csv")

# train = pd.read_csv(cwd+"/DATA/trial.csv")
# test = pd.read_csv(cwd+"/DATA/trial2.csv")

def fitmodel(train, test, verbose = 0, train_model = False):

    trainY = train['price']
    testY = test['price']

    trainX = train.drop(['price'], axis = 1)
    testX = test.drop(['price'], axis = 1)

    if train_model == True:
        print("Training model...\n")
        params = {"alpha": st.uniform()}

        ridge_model = Ridge()

        tmp = RandomizedSearchCV(ridge_model, params, n_jobs=-1, cv=10, verbose = verbose, n_iter = 25)
        tmp.fit(trainX, trainY)
        print("Optimised parameters: ")
        print(tmp.best_params_)
        ridge_opt = tmp.best_estimator_

    else:
        ridge_opt = Ridge(alpha=0.96295180239438127, copy_X=True, fit_intercept=True,
           max_iter=None, normalize=False, random_state=None, solver='auto',
           tol=0.001)

    print("Final model: ")
    print(ridge_opt)

    ridge_opt.fit(trainX, trainY)
    trainY_pred = ridge_opt.predict(trainX)
    testY_pred = ridge_opt.predict(testX)

    print("Performance metrics: \n")
    print("RMSE : %.4f (train) %.4f (test)" % (mean_squared_error(trainY, trainY_pred) ** .5, mean_squared_error(testY, testY_pred) ** .5))
    print("MAE : %.4f (train) %.4f (test)" % (mean_absolute_error(trainY, trainY_pred), mean_absolute_error(testY, testY_pred)))
    print("MedianAE : %.4f (train) %.4f (test)" % (median_absolute_error(trainY, trainY_pred), median_absolute_error(testY, testY_pred)))

    train_err = np.absolute(trainY - trainY_pred)/trainY
    train_err_mean = np.mean(train_err)
    test_err = np.absolute(testY - testY_pred)/testY
    test_err_mean = np.mean(test_err)

    print ("Mean Absolute Percentage Error: %.4f (train) %.4f (test)" % (train_err_mean, test_err_mean))
    print ("Median Absolute Percentage Error: %.4f (train) %.4f (test)" % (np.median(train_err), np.median(test_err)))

    th = [.01, .05, .1, .2, .3]
    train_err_vec = np.zeros(len(th))
    test_err_vec = np.zeros(len(th))

    for i in range(len(th)):
        train_err_vec[i] = np.sum((train_err < th[i] *1)) / len(train_err)
        test_err_vec[i] = np.sum((test_err < th[i] *1 )) / len(test_err)
        print("Absolute Percentage Error within %.2f: %.2f (train), %.2f (test)" % (th[i], train_err_vec[i], test_err_vec[i]))

fitmodel(train, test, train_model = False)

# # Printed Results
# Final model:
# Ridge(alpha=0.9629518023943813, copy_X=True, fit_intercept=True,
#    max_iter=None, normalize=False, random_state=None, solver='auto',
#    tol=0.001)
# Performance metrics:
#
# RMSE : 523669.7797 (train) 515727.6231 (test)
# MAE : 146387.3765 (train) 145897.8742 (test)
# MedianAE : 78943.8217 (train) 79093.7715 (test)
# Mean Absolute Percentage Error: 0.5539 (train) 0.5534 (test)
# Median Absolute Percentage Error: 0.3609 (train) 0.3608 (test)
# Absolute Percentage Error within 0.01: 0.02 (train), 0.02 (test)
# Absolute Percentage Error within 0.05: 0.08 (train), 0.08 (test)
# Absolute Percentage Error within 0.10: 0.15 (train), 0.15 (test)
# Absolute Percentage Error within 0.20: 0.30 (train), 0.30 (test)
# Absolute Percentage Error within 0.30: 0.43 (train), 0.43 (test)
