import sklearn
import imblearn
import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def feature_selection(df):
    # feature selection
    x = np.asanyarray(df[['Lon (deg)', 'Lat (deg)', 'Alt (m)']])
    y = np.asanyarray(df[['SEU']])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=4)
    return [x_train, y_train, x_test, y_test]


def data_undersampling(x_train, y_train):
    # training data sampling
    under_sample = ClusterCentroids(voting='soft')  # soft as we ourselves augmented the data
    x_train, y_train = under_sample.fit_resample(x_train, y_train)
    return x_train, y_train


def get_data(filename):
    df = load_data(filename)
    x_train, y_train, x_test, y_test = feature_selection(df)
    x_train, y_train = data_undersampling(x_train, y_train)
    print('Data Processing Completed')
    return x_train, y_train, x_test, y_test
