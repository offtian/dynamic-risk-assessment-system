"""
Author: Ollie Tian
Date: December, 2022
This script used for deploying trained model
"""
import os, sys, logging
import shutil

from constants import OUTPUT_FOLDER_PATH, OUTPUT_MODEL_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    logging.info("Deploying trained model to production")
    logging.info("Copying trainedmodel.pkl, ingestfiles.txt and latestscore.txt")
    shutil.copy(
        os.path.join(OUTPUT_FOLDER_PATH, "ingestedfiles.txt"), PROD_DEPLOYMENT_PATH
    )
    shutil.copy(
        os.path.join(OUTPUT_MODEL_PATH, "trainedmodel.pkl"), PROD_DEPLOYMENT_PATH
    )
    shutil.copy(
        os.path.join(OUTPUT_MODEL_PATH, "latestscore.txt"), PROD_DEPLOYMENT_PATH
    )


if __name__ == "__main__":
    logging.info("Running deployment.py")
    store_model_into_pickle()
