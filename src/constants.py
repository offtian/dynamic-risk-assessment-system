"""
Author: Ollie Tian
Date: December, 2022
This script used for storing all constant folder paths
"""
import os
import json

#############Load root directory paths
ROOT_DIR = os.path.abspath(os.getcwd())
# print(ROOT_DIR)

#############Load config.json and get input and output paths
with open(os.path.join(ROOT_DIR, "config.json"), "r") as f:
    config = json.load(f)

#############Load all relevant folder paths
INPUT_FOLDER_PATH = os.path.join(ROOT_DIR, config["input_folder_path"])
OUTPUT_FOLDER_PATH = os.path.join(ROOT_DIR, config["output_folder_path"])
TEST_DATA_PATH = os.path.join(ROOT_DIR, config["test_data_path"])
OUTPUT_MODEL_PATH = os.path.join(ROOT_DIR, config["output_model_path"])
PROD_DEPLOYMENT_PATH = os.path.join(ROOT_DIR, config["prod_deployment_path"])
