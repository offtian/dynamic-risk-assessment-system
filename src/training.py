"""
Author: Ollie Tian
Date: December, 2022
This script used for training ML model
"""
import pandas as pd
import pickle
import os, sys, logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from constants import OUTPUT_FOLDER_PATH, OUTPUT_MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################Function for training the model
def train_model():

    logging.info("Loading and preparing finaldata.csv")
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER_PATH, "finaldata.csv"))

    X = df.drop(["corporation", "exited"], axis=1)
    y = df["exited"]

    logging.info("Train, test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # use this logistic regression for training
    logging.info("Loading logistic regression model")
    clf = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=42,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    logging.info("Fitting X, y")
    clf.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Saving trained model")
    pickle.dump(clf, open(os.path.join(OUTPUT_MODEL_PATH, "trainedmodel.pkl"), "wb"))


if __name__ == "__main__":
    logging.info("Running training.py")
    train_model()
