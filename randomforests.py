import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from os.path import dirname, realpath
cwd = dirname(realpath(__file__))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

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
            "max_features": st.randint(5, 30),
            "min_samples_split": st.randint(2, 50),
            "min_samples_leaf": st.randint(1, 50),
        }
        rf_model = RandomForestRegressor(n_estimators = 5)
        tmp = RandomizedSearchCV(rf_model, params, n_jobs=-1, cv = 10, verbose = verbose, n_iter = 25)
        tmp.fit(trainX, trainY)
        print("Optimised parameters: ")
        print(tmp.best_params_)
        rf_opt = tmp.best_estimator_
        rf_opt.set_params(n_estimators=100, n_jobs = -1)
    else:
        rf_opt = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                   max_features=19, max_leaf_nodes=None, min_impurity_split=1e-07,
                   min_samples_leaf=20, min_samples_split=7,
                   min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                   oob_score=False, random_state=None, verbose=0, warm_start=False)

    print("Final model: ")
    print(rf_opt)

    rf_opt.fit(trainX, trainY)
    trainY_pred = rf_opt.predict(trainX)
    testY_pred = rf_opt.predict(testX)

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
