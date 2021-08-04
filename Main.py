import yaml
import logging
import re
import time


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import category_encoders as ce
from imblearn.over_sampling import SMOTE

from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
#Import models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, \
    recall_score, f1_score, classification_report, ConfusionMatrixDisplay, r2_score

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
    df.loc[:, config.get("object_cols")] = df.loc[:, config.get("object_cols")].astype("category")

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
    :return: train sets for regression and classification models
    """
    #Select features for regression model
    select_k_best = SelectKBest(score_func=f_regression, k=8).fit(x_train, y_train)
    best_cols = select_k_best.get_support(indices=True)
    x_train_regr = x_train.iloc[:, best_cols]

    #Select features for classification model
    select_k_best = SelectKBest(score_func=f_classif, k=8).fit(x_train, y_train)
    best_cols = select_k_best.get_support(indices=True)
    x_train_classif = x_train.iloc[:, best_cols]
    print(x_train.Pressure3pm.describe())
    print("Best features selected")
    return x_train_regr, x_train_classif


def tune_reg_models(x_train_regr, y_train):
    """
    Tuning hyper-parameters of regression estimators and selecting best estimators
    :param x_train_regr: train data (features) for regression models
    :param y_train: train data (target)
    :return: dictionary of best regression estimators
    """
    regr_models = {"lin_reg":LinearRegression(),
                  "log_reg":LogisticRegression()}
    regr_estimators = {}
    logging.info("Tuning regression models...\n")
    for estimator_name, estimator in regr_models.items():
        print("Training ", estimator_name)
        start = time.time()
        clf = HalvingGridSearchCV(estimator=estimator,
                                    param_grid=config.get(estimator_name),
                                    scoring="r2",
                                    verbose=5,
                                    n_jobs=-1,
                                    cv=10)
        clf.fit(x_train_regr, y_train)
        stop = time.time()
        logging.info(f"Training time: {stop - start}s")
        best_score = clf.best_score_
        regr_estimators[estimator_name] = clf.best_estimator_
        logging.info("{} best score: {}".format(estimator, best_score))
    print("Regression models tuned")
    return regr_estimators


def tune_classif_models(x_train_classif, y_train):
    """
    Tuning hyper-parameters of classification estimators and selecting best estimators
    :param x_train_classif: train data (features) for classification models
    :param y_train: train data (target)
    :return: dictionary of best classification estimators
    """
    classif_models = {"svm":SVC(),
                      "knn":KNeighborsClassifier(),
                      "decision_tree":DecisionTreeClassifier(),
                      "random_forest":RandomForestClassifier(),
                      "mlp":MLPClassifier()}

    logging.info("Tuning regression models...\n")
    classif_estimators = {}
    for estimator_name, estimator in classif_models.items():
        print("Training ", estimator_name)
        start = time.time()
        clf = HalvingGridSearchCV(estimator=estimator,
                                    param_grid=config.get(estimator_name),
                                    scoring="roc_auc",
                                    verbose=5,
                                    n_jobs=-1,
                                    cv=10)
        clf.fit(x_train_classif, y_train)
        stop = time.time()
        logging.info(f"Training time: {stop - start}s")
        best_score = clf.best_score_
        classif_estimators[estimator_name] = clf.best_estimator_
        logging.info("{} best score: {}".format(estimator, best_score))
    print("Classification models tuned")
    return classif_estimators


def evaluate_models(regr_estimators, classif_estimators,
                    x_train_regr, x_train_classif, x_test, y_train, y_test):
    """
    Retraining best ML models and evaluating them
    :param regr_estimators: dictionary of regression models
    :param classif_estimators: dictionary of classification models
    :param x_train_regr: regression train data (features)
    :param x_train_classif: classification train data (features)
    :param x_test: test data (features)
    :param y_train: train data (target)
    :param y_test: test data (target)
    :return:
    """
    evaluation_data = {}

    logging.info("Evaluating regression models: \n")
    for model_name, estimator in regr_estimators.items():
        estimator.fit(x_train_regr, y_train)
        predict = estimator.predict(x_test)
        predict_probability = estimator.predict_proba(x_test)[:, 1]

        logging.info("{}: \n".format(model_name))
        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict)
        recall = recall_score(y_test, predict)
        cm = confusion_matrix(y_test, predict)
        cr = classification_report(y_test, predict)
        f1 = f1_score(y_test, predict)
        roc_auc = roc_auc_score(y_test, predict_probability)
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))
        logging.info("Recall: {}".format(recall))
        logging.info("Confusion matrix: \n{}".format(cm))
        logging.info("Classification report: \n{}".format(cr))
        logging.info("F1 score: {}".format(f1))
        logging.info("ROC AUC score: {}\n".format(roc_auc))
        evaluation_data[model_name] = [accuracy, f1, roc_auc]

        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False,
                    annot_kws={'fontsize': 15}, yticklabels=['0', '1'],
                    xticklabels=['Predict 0', 'Predict 1'])
        fig.savefig("results/{}_confusion_matrix.png".format(model_name))

    logging.info("Evaluating classification models: \n")
    for model_name, estimator in classif_estimators.items():
        estimator.fit(x_train_classif, y_train)
        predict = estimator.predict(x_test)
        predict_probability = estimator.predict_proba(x_test)[:, 1]

        logging.info("{}: \n".format(model_name))
        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict)
        recall = recall_score(y_test, predict)
        cm = confusion_matrix(y_test, predict)
        cr = classification_report(y_test, predict)
        f1 = f1_score(y_test, predict)
        roc_auc = roc_auc_score(y_test, predict_probability)
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Precision: {}".format(precision))
        logging.info("Recall: {}".format(recall))
        logging.info("Confusion matrix: \n{}".format(cm))
        logging.info("Classification report: \n{}".format(cr))
        logging.info("F1 score: {}".format(f1))
        logging.info("ROC AUC score: {}\n".format(roc_auc))
        evaluation_data[model_name] = [accuracy, f1, roc_auc]

        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False,
                    annot_kws={'fontsize': 15}, yticklabels=['0', '1'],
                    xticklabels=['Predict 0', 'Predict 1'])
        fig.savefig("results/{}_confusion_matrix.png".format(model_name))

    df_evaluation = pd.DataFrame.from_dict(evaluation_data, orient="index",
                                           columns=["accuracy", "f1_score", "roc_auc_score"])
    df_evaluation.to_csv("results/evaluation.csv")
    print("Work complete!")


with open("Config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, filename="Results/logfile")

df  = load_dataset("weatherAUS.csv")
df = optimize_data(df)
df = fill_data(df)
df = scale_cat_data(df)
df = handle_outliers(df)
x_train, x_test, y_train, y_test = split_sample_data(df)
x_train, x_test = scale_data(x_train, x_test)
x_train_regr, x_train_classif  = select_features(x_train, y_train)
regr_estimators = tune_reg_models(x_train_regr, y_train)
classif_estimators = tune_classif_models(x_train_classif, y_train)
evaluate_models(regr_estimators, classif_estimators,
                x_train_regr, x_train_classif, x_test, y_train, y_test)