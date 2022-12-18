"""
Author: Ollie Tian
Date: December, 2022
This script used for producing confusion matrix report
"""
import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os, sys, logging
from constants import PROD_DEPLOYMENT_PATH, TEST_DATA_PATH, OUTPUT_MODEL_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


##############Function for reporting
def plot_confusion_matrix(file_name="confusionmatrix.png"):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # read the deployed model and a test dataset, calculate predictions
    logging.info("Loading deployed model")
    clf = pickle.load(
        open(os.path.join(PROD_DEPLOYMENT_PATH, "trainedmodel.pkl"), "rb")
    )
    
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    X = test_df.drop(['corporation', 'exited'], axis=1)
    y_true = test_df['exited']
    
    y_pred = clf.predict(X)
    
    logging.info("Plotting confusion matrix...")
    disp = metrics.ConfusionMatrixDisplay.from_estimator(clf, X, y_true,  cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion matrix - Exited or not")
    
    logging.info(f"Saving confusion matrix into {OUTPUT_MODEL_PATH}")
    plt.savefig(os.path.join(OUTPUT_MODEL_PATH, file_name))
    
    return True




if __name__ == '__main__':
    plot_confusion_matrix(file_name="confusionmatrix.png")
