"""
Author: Ollie Tian
Date: December, 2022
This script used for setting up a series of API Endpoint
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import re
import subprocess
from constants import PROD_DEPLOYMENT_PATH, TEST_DATA_PATH, OUTPUT_MODEL_PATH
import diagnostics as diagnostics

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


prediction_model = None


@app.route('/')
def index():
    return "Welcome to use dynamic risk assessment system!"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Prediction endpoint that loads data given the file path
    and calls the prediction function in diagnostics.py
    Returns:
        json: model predictions
    """
    filepath = request.get_json()['filepath']
    
    df = pd.read_csv(filepath)
    
    #call the prediction function you created in Step 3
    preds = diagnostics.model_predictions(df.drop(['corporation', 'exited'], axis=1))
    
    return jsonify(preds.tolist()) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring(): 
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model
    Returns:
        str: model f1 score
    """       
    #check the score of the deployed model
    
    output = subprocess.run(['python', 'scoring.py'],
                            capture_output=True).stdout
    output = re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    return output #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():
    """
    Summary statistics endpoint that calls dataframe summary
    function from diagnostics.py
    Returns:
        json: summary statistics
    """        
    #check means, medians, and modes for each column
    return jsonify(diagnostics.dataframe_summary()) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():
    """
    Diagnostics endpoint thats calls missing_percentage, execution_time,
    and outdated_package_list from diagnostics.py
    Returns:
        dict: missing percentage, execution time and outdated packages
    """
    #check timing and percent NA values
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()
    
    outputs = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }
    return jsonify(outputs) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
