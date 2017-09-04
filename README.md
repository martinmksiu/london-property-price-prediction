## London Property Price Prediction

The code hosted in this folder is for my dissertation at my university. The research uses Ridge Regression, Gradient Boosting and Random Forests to predict property prices in London. Needless to say, the three Python files are the codes that train the regression model for each regression technique. 

### Packages required
- pandas
- numpy
- sklearn
- xgboost
- matplotlib.pyplot

### Usage
Before running the code, download the csv data as stated in the README file in the Data directory. The format for all files are similar to the one for Ridge Regression shown below:

```python
fitmodel(train, test, verbose = 0, train_model = False)
```

`verbose` controls the verbosity in `RandomizedSearchCV`. The higher the integer, the more messages.

`train_model` indicates whether to train a new model for the data. If `train_model = False`, it uses the model I trained using training data in `train.csv`.

Gradient Boosting and Random Forests have the additional argument `plot_graph` to plot the feature importance graph. By default:

```python
fitmodel(train, test, verbose = 0, train_model = False, plot_graph = False)
```