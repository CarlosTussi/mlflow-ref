# Databricks notebook source
# MAGIC %md This notebook is slightly modified version of `MLflow Training Tutorial` from [MLflow examples](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine).
# MAGIC
# MAGIC It predicts the quality of wine using [sklearn.linear_model.ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
# MAGIC This is a base code and will be modified further during the `Databricks: Reproducible experiments with MLflow and Delta Lake` tutorial.
# MAGIC
# MAGIC Attribution
# MAGIC * The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# MAGIC * P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# MAGIC * Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.



'''
Run the server before this module:

1) mlflow server --host 127.0.0.1 --port 8080


2)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")
# Beforoe the run. Needs to be defined only once.
mlflow.autolog()

# For each set of parameters we do a new run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)
    #mlflow.autolog()

    ###### ML CODE HERE #######
    ###### ML CODE HERE #######
    ###### ML CODE HERE #######
    ###### ML CODE HERE #######
    ###### ML CODE HERE #######

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic Elastic Model")
    
    # Infer the model signature
    signature = infer_signature(train_x, lr.predict(train_x))   # Define before in the ML Code or before
    
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,                                           # Define before in the ML Code or before
        artifact_path="wine_model",                            # Custom name
        signature= signature,
        input_example=train_x,
        registered_model_name= type(lr).__name__,               # This will log the correct model being executed
    )
    
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    
    
    
    
    
    predictions = loaded_model.predict(test_x)

'''

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import logging

import mlflow
from mlflow.models import infer_signature

# Define the model hyperparameters

list_params = [
        {
    "alpha" : 0.5, 
    "l1_ratio" : 0.5, 
    "random_state" : 42
        },
        {
    "alpha" : 0.7, 
    "l1_ratio" : 0.1, 
    "random_state" : 77
        }
              ]

params = {
    "alpha" : 0.5, 
    "l1_ratio" : 0.5, 
    "random_state" : 42
}


# Read the wine-quality csv file from the URL
csv_url =\
'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
    data = pd.read_csv(csv_url, sep=';')
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s", e)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")
# Beforoe the run. Needs to be defined only once.
mlflow.autolog()

# For each set of parameters we do a new run
for params in list_params:
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)
        #mlflow.autolog()
    
        ###### ML CODE HERE #######
        def eval_metrics(actual, pred):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
    
    
        warnings.filterwarnings("ignore")
        np.random.seed(40)
    
        # Execute ElasticNet
        lr = ElasticNet(**params)
        lr.fit(train_x, train_y)
    
        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    
        ###### ML CODE HERE #######
    

        # Log the loss metric
        # If autolog is enabled we dont need to defined these logs manually.
        #mlflow.log_metric("r2", r2)
        #mlflow.log_metric("rsme", r2)
        #mlflow.log_metric("mae", r2)
    
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic Elastic Model")
    
        # Infer the model signature
        signature = infer_signature(train_x, lr.predict(train_x))
    
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="wine_model",
            signature= signature,
            input_example=train_x,
            registered_model_name= type(lr).__name__,    # This will log the correct model being executed
        )
    
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    
    
    
    
    
    predictions = loaded_model.predict(test_x)