"""
Author: Ollie Tian
Date: December, 2022
This script used for ingesting data
"""
import logging

import os, sys
import glob

import pandas as pd

from datetime import datetime
from constants import INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#############Function for data ingestion
def merge_multiple_dataframe(file_pattern="csv"):
    # check for datasets, compile them together, and write to an output file

    logging.info(f"Reading all {file_pattern} files from {INPUT_FOLDER_PATH}")
    all_files = glob.glob(os.path.join(INPUT_FOLDER_PATH, f"*.{file_pattern}"))

    if file_pattern == "csv":
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    else:
        print("This file extension is not supported yet.")

    logging.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("Saving ingested metadata")
    with open(os.path.join(OUTPUT_FOLDER_PATH, "ingestedfiles.txt"), "w") as file:
        file.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(all_files))

    logging.info("Saving ingested data")
    df.to_csv(os.path.join(OUTPUT_FOLDER_PATH, "finaldata.csv"), index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()
