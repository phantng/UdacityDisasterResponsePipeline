import os
import re
import sys
import pickle

import pandas as pd
import numpy as np
import sqlalchemy

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from preprocess_utils import tokenize


def load_data(path: str = "../data/DisasterResponse.db"):
    """
    Loads table from path to SQLite database
    :param path: str path to SQLite file
    :return: tuple of (features, targets)
    """
    # read data from SQLite database into a dataframe
    engine = sqlalchemy.create_engine(os.path.join("sqlite:///", path))
    df = pd.read_sql_table("DisasterResponseTable", con=engine)

    # separate data into features and targets
    x = df["message"].copy()
    y = df.iloc[:, 4:]
    return x, y, y.columns.tolist()


def build_model():
    """Build a model using sklearn and custom tokenizer.
    Returns an sklearn Estimator
    """
    # instantiate pipeline
    vectorizer = CountVectorizer(tokenizer=tokenize)
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=42, criterion="gini"))
    pipeline = Pipeline([("vectorizer", vectorizer), ("clf", clf)])

    # search for best parameters among specified
    param_grid = {"clf__estimator__max_depth": [2, 4]}
    model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, cv=4, refit=True,
                         return_train_score=True, verbose=1)
    return model


def evaluate_model(model, x_test, y_test, category_names: list[str]):
    """
    Evaluate a model and print out the classification report
    :param model: sklearn classifier
    :param x_test: test data
    :param y_test: test targets
    :param category_names: names of columns for test targets
    :return: None
    """
    y_pred = model.predict(x_test)
    mlb = MultiLabelBinarizer().fit(y_test)
    report = classification_report(mlb.transform(y_test), mlb.transform(y_pred), target_names=category_names)
    print(report)


def save_model(model, model_filepath: str):
    """
    Save a model to a local location.
    :param model: sklearn classifier
    :param model_filepath: path including file name of classifier to pickle
    :return: None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        # download prerequisites
        nltk.download("wordnet")  # download for lemmatization
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("omw-1.4")

        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")
    else:
        print("Please provide the filepath of the disaster messages database "
              "as the first argument and the filepath of the pickle file to "
              "save the model to as the second argument. \n\nExample: python "
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
