# Disaster Response Project

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Data](#data)
4. [Modelling](#modelling)

## Introduction

This repository contains the code for the project Disaster Response Pipeline on Udacity.

1. Run the following commands to set up your database, model, and start the webapp:

    1. To run ETL pipeline that cleans data and stores said data to a SQLite database:

       `python path/to/process_data.py path/to/message.csv path/to/disaster_categories.csv output_path/to/DisasterResponse.db`

    2. To run ML pipeline that trains a classifier and saves to a pickle file:

       `python path/to/train_classifier.py path/to/DisasterResponse.db output_path/to/classifier.pkl`

    3. To start your web app (note this requires the previous two steps were done sequentially):

       `python path/to/run.py`

2. Go to the address you've registered with Flask to see the webapp, by default it's "0.0.0.0".

## Requirements

```
Flask==2.1.2
joblib==1.1.1
nltk==3.7
numpy==1.22.4
pandas==1.4.2
plotly==5.11.0
scikit_learn==1.2.0
SQLAlchemy==1.4.37
```

## Data

You can find the data as well as the resulting database post preprocessing here:

* [**process_data.py**](data/process_data.py): Python ETL script, can configure path to input files and output database.
* [**DisasterResponse.db**](data/DisasterResponse.db): processed SQLite database.
* [**categories.csv**](data/categories.csv): dataset containing the categorical information about disaster responses.
* [**messages.csv**](data/messages.csv): dataset containing the actual messages in disaster responses.

Data was preprocessed as follows:

1. Merge the messages dataset and the categories dataset based on id.
2. Category was exploded into individual target columns and values are binarized.
3. The original "category" column is then dropped.
4. Duplicates based on "message" column are removed.
5. Data is then inserted into a SQLite database at the chosen local path.

## Modelling

* [**Trained model - clf.pkl**](models/clf.pkl): sklearn estimator in pickle format.
* [**train_classifier.py**](models/train_classifier.py): Python file for model training.
* [**utils.py**](models/utils.py): Utility model for defining tokenizer and scorer function. Meant to be imported in the
  training script to work with pickle.
* [**result.csv**](models/result.csv): Optional model evaluation output, where you can view the resulting trained
  model's performance.

The modelling process is roughly as follows:

1. Messages are tokenized into unique words.
    1. Data is first converted to lowercase.
    2. Stopwords and punctuations are removed from the data.
    3. Words are also lemmatized to reduce words to their roots with context.
    4. Tokenizer is then used in TfidfVectorizer as part of the pipeline
2. Trained and validated a machine learning pipeline using TfidfVectorizer from step 1, a RandomForestClassifier, and
   sklearn's Pipeline.
    1. Multi-output is supported by default with RandomForestClassifier, although a scorer function was also
       implemented.
    2. Split the data into training and test sets in 80-20 proportion.
    3. Then performed hyperparameter tuning with KFold cross-validation to find the best hyperparameters for predicting
       disaster response category.
    4. Models are then pickled to location specified when running the script as path/to/clf.pkl for later use in the web
       app.
