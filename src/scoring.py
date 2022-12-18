"""
Author: Ollie Tian
Date: December, 2022
This script used for ML model scoring
"""
import pandas as pd
import pickle
import os, sys, logging
from sklearn import metrics

from constants import TEST_DATA_PATH, OUTPUT_MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################Function for model scoring
def score_model(test_file_path, model_path, save_results=False):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    logging.info(f"Loading {test_file_path}")
    test_df = pd.read_csv(test_file_path)

    logging.info("Loading trained model")
    clf = pickle.load(open(model_path, "rb"))

    y_test = test_df["exited"]
    X_test = test_df.drop(["corporation", "exited"], axis=1)

    logging.info("Predicting on new data")
    y_pred = clf.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"f1 score = {f1}")

    logging.info("Saving scores to text file")
    if save_results:
        with open(os.path.join(OUTPUT_MODEL_PATH, "latestscore.txt"), "w") as file:
            file.write(f"f1 score = {f1}")
    
    return f1

if __name__ == "__main__":
    logging.info("Running scoring.py")
    test_file_path = os.path.join(TEST_DATA_PATH, "testdata.csv")
    model_path = os.path.join(OUTPUT_MODEL_PATH, "trainedmodel.pkl")
    score_model(test_file_path, model_path, save_results=False)
