import yaml
import logging
import re

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import category_encoders as ce
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, \
    recall_score, f1_score, classification_report, ConfusionMatrixDisplay

def load_dataset(csv):
    """
    Loading data from .csv file into pandas DataFrame
    :param csv: path to .csv file
    :return: DataFrame
    """
    df = pd.read_csv(csv)
    print("Data loaded")
    return df


def optimize_data(df):
    """
    Optimizing data by changing data type, dropping certain columns
    :param df: Initial DataFrame
    :return: Optimized DataFrame with less columns
    """
    #Changing column data type to category
    #cat_cols = df.select_dtypes(include="object")
    #df[cat_cols] = df[cat_cols].astype("category")

    #Dropping uninformative and redundant columns
    df.drop(columns=config.get("cols_to_drop"), axis=1, inplace=True)
    print("Data optimized")

    return df


def fill_data(df):
    """
    Filling missing data
    :param df: Initial DataFrame
    :return: DataFrame with no missing values
    """
    #Filling NaNs in columns with a lot of NaN values
    for col in config.get("nan_cols"):
        df[col] = df[col].fillna(method="bfill").fillna(value=df[col].mean())

    #Filling the rest of columns with mean values
    df.fillna(df.mean(), inplace=True)

    #Filling NaNs in RainToday, RainTomorrow and replacing values
    for col in config.get("rain_cols"):
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col].replace(["Yes", "No"], [1, 0], inplace=True)

    #Filling NaNs in categorical wind columns
    for col in config.get("wind_cols"):
        df[col] = df[col].fillna(method="bfill").fillna(df[col].mode()[0])
    print("Missing values filled")

    return df


def scale_cat_data(df):
    """
    Binning and scaling data with TargetEncoder in wind columns
    :param df: Initial DataFrame
    :return: DataFrame with scaled wind data
    """
    #Binning data
    for col in config.get("wind_cols"):
        for direction in config.get("wind_direction"):
            mask = df[col].str.extract(fr"(^{direction}\w?\w?)").index
            df[col].loc[mask] = direction
    #Initializing TargetEncoder
    encoder = ce.TargetEncoder().fit(df.loc[:, config.get("wind_cols")],
                                     df.loc[:, config.get("target")])
    encoded_cols = encoder.transform(df.loc[:, config.get("wind_cols")])
    df.drop(columns=config.get("wind_cols"), inplace=True, axis=1)
    df = df.join(encoded_cols)
    print("Categorical data encoded")

    return df


def handle_outliers(df):
    """
    Deleting outliers using IQR
    :param df: Initial DataFrame
    :return: DataFrame without outliers
    """
    for col in config.get("outlier_cols"):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1
        mask = df[col].loc[(df[col] < (q1 - 1.5 * IQR)) | (df[col] > (q3 + 1.5 * IQR))]
        df.drop(labels=mask.index, inplace=True)
    print("Outliers deleted")

    return df


def split_sample_data(df):
    """
    Splitting data into train and test portions and oversampling train data
    :param df: DataFrame
    :return: splits of train and test data
    """
    #Splitting data into train and test
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != config.get("target")],
                                                        df[config.get("target")],
                                                        test_size=0.2,
                                                        random_state=42)
    print("Data split")

    #Oversampling train data
    smote = SMOTE(n_jobs=-1)
    x_train, y_train = smote.fit_resample(df.loc[:, df.columns != config.get("target")],
                                          df[config.get("target")])
    print("Data sampled")

    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    """
    Scaling numerical data with RobustScaler and MinMaxScaler
    :param x_train: Initial train set
    :param x_test: Initial test set
    :return: Scaled train and test data
    """
    #Scaling data containing outliers
    robust_scaler = RobustScaler(quantile_range=(1, 99))
    x_train.loc[:, config.get("robust_cols")] = \
        robust_scaler.fit_transform(x_train.loc[:, config.get("robust_cols")])
    x_test.loc[:, config.get("robust_cols")] = \
        robust_scaler.transform(x_test.loc[:, config.get("robust_cols")])

    #Scaling data with MinMaxScaler
    min_max = MinMaxScaler()
    x_train.loc[:, config.get("min_max_cols")] = \
        min_max.fit_transform(x_train.loc[:, config.get("min_max_cols")])
    x_test.loc[:, config.get("min_max_cols")] = \
        min_max.transform(x_test.loc[:, config.get("min_max_cols")])
    print("Data scaled")

    return x_train, x_test


def select_features(x_train, y_train):
    """
    Selecting best features with SelectKBest
    :param x_train: train features
    :param y_train: train target
    :return: selected train features
    """

    knn = KNeighborsClassifier()
    sfs = SequentialFeatureSelector(knn, n_features_to_select=8)
    sfs.fit(x_train, y_train)
    print("test")
    best_cols = sfs.get_support(indices=True)
    x_train = x_train.iloc[:, best_cols]
    x_train.head()

    # select_k_best = SelectKBest(score_func=f_classif, k=8).fit(x_train, y_train)
    # x_train = select_k_best.transform(x_train)
    # best_cols = select_k_best.get_support(indices=True)
    # x_train = x_train.iloc[:, best_cols]

    return x_train


def tune_models(x_train, y_train):
    """
    Training various ML models and tuning their parameters
    :param x_train: training features
    :param y_train: training target
    :return:
    """

    pass

with open("Config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

df  = load_dataset("weatherAUS.csv")
df = optimize_data(df)
df = fill_data(df)
df = scale_cat_data(df)
df = handle_outliers(df)
x_train, x_test, y_train, y_test = split_sample_data(df)
x_train, x_test = scale_data(x_train, x_test)
x_train = select_features(x_train, y_train)