import pandas as pd
import numpy as np
import scipy.stats as st
from os.path import dirname, realpath
cwd = dirname(realpath(__file__))
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

print("Loading data...\n")
train = pd.read_csv(cwd+"/Data/train.csv")
test = pd.read_csv(cwd+"/Data/test.csv")

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
           max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    print("Final model: ")
    print(ridge_opt)

    ridge_opt.fit(trainX, trainY)
    trainY_pred = ridge_opt.predict(trainX)
    testY_pred = ridge_opt.predict(testX)
    print("Performance metrics: \n")
    print("RMSE : %.4f (train) %.4f (test)" % (mean_squared_error(trainY, trainY_pred) ** .5,
                                                mean_squared_error(testY, testY_pred) ** .5))
    print("MAE : %.4f (train) %.4f (test)" % (mean_absolute_error(trainY, trainY_pred),
                                                mean_absolute_error(testY, testY_pred)))
    print("MedianAE : %.4f (train) %.4f (test)" % (median_absolute_error(trainY, trainY_pred),
                                                median_absolute_error(testY, testY_pred)))
    train_err = np.absolute(trainY - trainY_pred)/trainY
    test_err = np.absolute(testY - testY_pred)/testY
    print ("Mean Absolute Percentage Error: %.4f (train) %.4f (test)" %
                                                        (np.mean(train_err), np.mean(test_err)))
    print ("Median Absolute Percentage Error: %.4f (train) %.4f (test)" %
                                                        (np.median(train_err), np.median(test_err)))

    th = [.01, .05, .1, .2, .3]
    train_err_vec = np.zeros(len(th))
    test_err_vec = np.zeros(len(th))
    for i in range(len(th)):
        train_err_vec[i] = np.sum((train_err < th[i] *1)) / len(train_err)
        test_err_vec[i] = np.sum((test_err < th[i] *1 )) / len(test_err)
        print("Absolute Percentage Error within %.2f: %.2f (train), %.2f (test)"
                                                        % (th[i], train_err_vec[i], test_err_vec[i]))

fitmodel(train, test, train_model = False)
