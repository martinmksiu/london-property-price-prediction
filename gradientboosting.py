import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from os.path import dirname, realpath
cwd = dirname(realpath(__file__))
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from xgboost import XGBRegressor

print("Loading data...\n")
train = pd.read_csv(cwd+"/Data/train.csv")
test = pd.read_csv(cwd+"/Data/test.csv")

def fitmodel(train, test, verbose = 0, train_model = False, plot_graph = False):
    trainY = train['price']
    testY = test['price']
    trainX = train.drop(['price'], axis = 1)
    testX = test.drop(['price'], axis = 1)

    if train_model == True:
        print("Training model...")
        params = {
        "max_depth": st.randint(3, 40),
        "colsample_bytree": st.beta(10, 1)  ,
        "subsample": st.beta(10, 1)  ,
        "gamma": st.uniform(0, 10),
        "min_child_weight": st.expon(0, 50),
        }
        gboost = XGBRegressor(n_estimators = 5, learning_rate = .2)
        tmp = RandomizedSearchCV(gboost, params, cv = 10, n_jobs = -1, verbose = verbose, n_iter = 25)
        tmp.fit(trainX, trainY)
        print("Optimised parameters: ")
        print(tmp.best_params_)
        gboost_opt = tmp.best_estimator_
        gboost_opt.set_params(n_estimators=100, learning_rate = .1, n_jobs = -1)
    else:
        gboost_opt = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=0.87219466652443045, gamma=7.0610396795642156,
        learning_rate=0.1, max_delta_step=0, max_depth=23,
        min_child_weight=13.539302225736687, missing=None, n_estimators=100,
        n_jobs=-1, nthread=None, objective='reg:linear', random_state=0,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        silent=True, subsample=0.95498622807161138)

    print("Final model: ")
    print(gboost_opt)

    gboost_opt.fit(trainX, trainY)
    trainY_pred = gboost_opt.predict(trainX)
    testY_pred = gboost_opt.predict(testX)
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

    if plot_graph == True:
        feat_imp = pd.Series(gboost_opt.feature_importances_, list(trainX)).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()

fitmodel(train, test, train_model = False, plot_graph = False)
