"""
Author: Ollie Tian
Date: December, 2022
This script used for running full process, testing model drift, 
and re-train and re-deployment of model. 
"""
import logging, sys, os
import subprocess

import ingestion as ingestion
import training as training
import scoring as scoring
import deployment as deployment
import diagnostics as diagnostics
import reporting as reporting
from constants import *


logging.basicConfig(filename='./logs.log', level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(testing_mode=False):

    # checking if directory for model exists
    if not os.path.isdir(OUTPUT_MODEL_PATH):
        os.mkdir(OUTPUT_MODEL_PATH)
        logger.info(f'Creation of directory {OUTPUT_MODEL_PATH} where the model will be saved')


    ##################Check and read new data
    # first, read ingestedfiles.txt
    try: 
        with open(os.path.join(PROD_DEPLOYMENT_PATH, 'ingestedfiles.txt')) as f:
            ingested_files = f.read().split('\n')
            
        ingested_files = [f for f in ingested_files if f]  # remove any empty string
        first_implementation = False
    except FileNotFoundError:
        logger.info('No file has been ingested yet in production')
        logger.info('This is the first time the production model is deployed')
        ingested_files = []
        first_implementation = True
    
    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    filenames = next(os.walk(INPUT_FOLDER_PATH), (None, None, []))[2]  # [] if no file
    new_data = [file for file in filenames if file not in ingested_files]   


    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if new_data:
        logger.info('There are new data, we need to ingest them.')
        ingestion.merge_multiple_dataframe()
    elif not testing_mode:
        logger.info('No new file found, stop the process here.')
        exit()
    else:
        logger.info('No new file found.')
        logger.info('as we are in testing mode process continue. Should stop in production')

    ##################Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if first_implementation:
        model_drift = False
    else:
        # check whether the score from the deployed model is different from the score from the model that uses the
        # newest ingested data
        logger.info('checking for model drift using newly ingested data')
        with open(os.path.join(PROD_DEPLOYMENT_PATH, 'latestscore.txt'), 'r') as f:
            latest_score = float(f.read().split(" ")[-1])
            print("latest_score = ", latest_score)
        new_file_path = os.path.join(OUTPUT_FOLDER_PATH, 'finaldata.csv')
        model_path = os.path.join(PROD_DEPLOYMENT_PATH, "trainedmodel.pkl")
        new_score = scoring.score_model(new_file_path, model_path)
        model_drift = True if new_score > latest_score else False

    ##################Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if not model_drift and not testing_mode and not first_implementation:
        logger.info('No drift found, process stop here.')
        exit()
    elif first_implementation and not testing_mode:
        logger.info("First deployment in production of the model.")
    elif not testing_mode:
        logger.info('Model drift found we need to train and deploy a new model')
    else:
        logger.info('No drift found')
        logger.info('as we are in testing mode process continue. Should stop in production')

    # Re-training
    training.train_model()

    ##################Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()
    
    ##################Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model
    reporting.plot_confusion_matrix(file_name="confusionmatrix2.png")
    logger.info('Running diagnostics and reporting: execute apicalls.py')
    subprocess.run(['python3', 'src/apicalls.py'])

if __name__ == '__main__':
    print('starting')
    main(testing_mode=False)