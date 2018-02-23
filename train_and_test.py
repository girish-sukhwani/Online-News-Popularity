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


if __name__ == '__main__':

    import pandas as pd

    # Read data from a file and split into data and labels.
    path_to_file = 'OnlineNewsPopularity/OnlineNewsPopularity.csv'
    data = pd.read_csv(path_to_file, header=0).drop('url', axis=1)
    labels = pd.Series(data.pop(' shares'))

    # Split dataset into train and test sets.
    x_train, x_test, y_train, y_test = split_data(data, labels)
