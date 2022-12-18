"""
Author: Ollie Tian
Date: December, 2022
This script used for model and data diagnostics
"""
import pandas as pd
import timeit, subprocess
import os, sys, logging
import json, pickle
from typing import List, Dict

from constants import (
    OUTPUT_FOLDER_PATH,
    TEST_DATA_PATH,
    OUTPUT_MODEL_PATH,
    PROD_DEPLOYMENT_PATH,
    ROOT_DIR
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


##################Function to get model predictions
def model_predictions(X: pd.DataFrame) -> List:
    """Loads deployed model to predict on data provided

    Args:
        X (pd.DataFrame): Dependent variables

    Returns:
        list: Model predictions
    """
    # read the deployed model and a test dataset, calculate predictions
    logging.info("Loading deployed model")
    clf = pickle.load(
        open(os.path.join(PROD_DEPLOYMENT_PATH, "trainedmodel.pkl"), "rb")
    )

    logging.info("Running predictions on data")
    y_pred = clf.predict(X)

    # This list should have the same length as the number of rows in the input dataset.
    assert len(y_pred) == len(X)

    return y_pred  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary() -> Dict:
    """Loads finaldata.csv and calculates mean, median and std
    on all numerical data

    Returns:
        dict: A set of dictionary contains column name, mean, median and std.
    """

    # calculate summary statistics here
    logging.info("Loading and preparing finaldata.csv")
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER_PATH, "finaldata.csv"))
    df = df.select_dtypes("number")

    logging.info("Calculating statistics for data")
    statistics_dict = {}
    for col in df.columns:

        statistics_dict[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
        }

    return statistics_dict  # return value should be a list containing all summary statistics


##################Function to get missing percentage
def missing_percentage() -> Dict:
    """Calculates percentage of missing data for each column in
    finaldata.csv

    Returns:
        dict: Each dict contains column name and percentage
    """

    # calculate missing percentage here
    logging.info("Loading and preparing finaldata.csv")
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER_PATH, "finaldata.csv"))

    logging.info("Calculating missing percentage for data")
    missing_dict = {
        col: {"percentage": perc}
        for col, perc in zip(df.columns, df.isna().sum() / df.shape[0] * 100)
    }

    return (
        missing_dict  # return value should be a list containing all missing percentage
    )


##################Function to get timings
def execution_time() -> List:
    """Measures execution time for ingestion.py and training.py

    Returns:
        list: list of running time for ingestion.py and training.py
    """
    # calculate timing of training.py and ingestion.py
    logging.info("Calculating time for ingestion.py")
    ingestion_starttime = timeit.default_timer()
    _ = subprocess.run(["python", "ingestion.py"], capture_output=True)
    ingestion_timing = timeit.default_timer() - ingestion_starttime

    logging.info("Calculating time for training.py")
    training_starttime = timeit.default_timer()
    _ = subprocess.run(["python", "training.py"], capture_output=True)
    training_timing = timeit.default_timer() - training_starttime

    return [
        ingestion_timing,
        training_timing,
    ]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list() -> List:
    """Check dependencies status from requirements.txt file using pip-outdated
    which checks each package status if it is outdated or not

    Returns:
        List: stdout of the pip-outdated command
    """
    # get a list of
    logging.info("Checking outdated dependencies")
    dependencies = subprocess.run(
        f"pip-outdated {ROOT_DIR}/requirements.txt",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        shell=True
    )

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans("", "", " \t\r"))
    dep = dep.split("\n")
    
    # print(dep)
    # dep = [dep[3]] + dep[5:-3]
    # dep = [s.split("|")[1:-1] for s in dep]
    return dep


if __name__ == "__main__":
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    X = test_df.drop(['corporation', 'exited'], axis=1)

    print("Model predictions on testdata.csv:",
          model_predictions(X), end='\n\n')
    print("Summary statistics")
    print(json.dumps(dataframe_summary(), indent=4), end='\n\n')

    print("Missing percentage")
    print(json.dumps(missing_percentage(), indent=4), end='\n\n')

    print("Execution time")
    print(json.dumps(execution_time(), indent=4), end='\n\n')

    print("Outdated Packages")
    dependencies = outdated_packages_list()
    # for row in dependencies:
    #     print('{:<20}{:<10}{:<10}{:<10}'.format(*row))
    print(dependencies)