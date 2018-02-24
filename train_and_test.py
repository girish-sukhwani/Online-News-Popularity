def split_data(data, labels):
    '''Splits data into train and test data and labels.

    Args:
      data (Pandas DataFrame): A Pandas DataFrame containing the
                               dataset.
      labels (Pandas Series): A Pandas Series containing the labels.

    Returns:
      An array of Pandas DataFrames and Series containing the data
      after the split in the order `train data`, `test data`,
      `train labels` and `test labels`.
    '''

    from sklearn.model_selection import train_test_split

    return train_test_split(data,
                            labels,
                            test_size=0.33,
                            random_state=390142)

def grid_search(classifier, x_train, y_train):
    '''Performs grid search on train data and returns the best model.

    Args:
      classifier (object): An XGBoost object.
      x_train (Pandas DataFrame): A Pandas DataFrame containing the
                                  training data.
      y_train (Pandas Series): A Pandas Series containing the
                               training labels.

    Return:
      A GridSearchCV object containing the model with the best score.
    '''

    from sklearn.model_selection import GridSearchCV

    params = {
              'max_depth': [3, 6, 9],
              'n_estimators': [50, 100],
              'booster': ['gbtree', 'dart'],
              'learning_rate': [0.001, 0.01, 0.1],
              'gamma': [1, 10, 100],
              'subsample': [0.5, 0.7, 0.9]
    }

    grid_search_obj = GridSearchCV(classifier, params, cv=5, n_jobs=-1, verbose=1)
    grid_search_obj.fit(x_train, y_train)

    print('The following is the best parameter setting for this problem:')
    print(grid_search_obj.best_params_)

    print('Training score on the best estimator: {}'
          .format(grid_search_obj.best_score_))

    return grid_search_obj.best_estimator_

if __name__ == '__main__':

    import pandas as pd
    from xgboost import XGBRegressor

    # Read data from a file and split into data and labels.
    path_to_file = 'OnlineNewsPopularity/OnlineNewsPopularity.csv'
    data = pd.read_csv(path_to_file, header=0).drop('url', axis=1)
    labels = pd.Series(data.pop(' shares'))

    # Split dataset into train and test sets.
    x_train, x_test, y_train, y_test = split_data(data, labels)

    # Create an XGBRegressor object
    clf = XGBRegressor(objective='reg:gamma', n_jobs=-1, random_state=241093)

    # Perform Grid Search on a paramter grid
    best_clf = grid_search(clf, x_train, y_train)
