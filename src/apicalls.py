"""
Author: Ollie Tian
Date: December, 2022
This script used to call the APIs and generate a report file that includes
Model predictions and scores on test data
"""
import requests
import logging, os, sys
import pandas as pd

from constants import *


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Specify a URL that resolves to your workspace
URL = "http://192.168.1.104:8000/"

# Call each API endpoint and store the responses
logging.info(
    f"Post request /prediction for {os.path.join(TEST_DATA_PATH, 'testdata.csv')}"
)
response_pred = requests.post(
    f"{URL}/prediction",
    # header={"Content-Type": "application/json"},
    json={"filepath": os.path.join(TEST_DATA_PATH, "testdata.csv")},
).text  # put an API call here

# scoring
logging.info("Get request /scoring")
response_scor = requests.get(f"{URL}/scoring").text  # put an API call here

# summarystats
logging.info("Get request /summarystats")
response_stat = requests.get(f"{URL}/summarystats").text

# diagnostics
logging.info("Get request /diagnostics")
response_diag = requests.get(f"{URL}/diagnostics").text
response_diag = eval(response_diag)
# combine all API responses
# combine all API responses
responses = "-" * 50 + "\n"
responses += " " * 10 + "** Model reporting **\n"
responses += "-" * 50 + "\n\n"
responses += " " * 10 + "Predictions:\n\n"
responses += response_pred + "\n"
responses += "-" * 50 + "\n"
responses += " " * 10 + "F1 score:\n\n"
responses += response_scor + "\n"
responses += "-" * 50 + "\n"
responses += " " * 10 + "Statistics:\n\n"
df_stats = pd.DataFrame(eval(response_stat)).T
responses += df_stats.to_string() + "\n"
responses += "-" * 50 + "\n"
responses += " " * 10 + "Diagnostics:\n\n"
responses += "missing data per column (%):\n"
print(type(response_diag))
responses += (
    str(response_diag['missing_percentage']) + "\n\n"
)
responses += "Ingestion and training execution time:\n"
responses += str(response_diag["execution_time"]) + "\n\n"
responses += "outdated packages:\n"
responses += str(response_diag["outdated_packages"]) + "\n\n"
responses += "-" * 50 + "\n"

report_path = os.path.join(OUTPUT_MODEL_PATH, "apireturns.txt")  # combine reponses here

# write the responses to your workspace
with open(report_path, "w") as f:
    f.write(responses)
