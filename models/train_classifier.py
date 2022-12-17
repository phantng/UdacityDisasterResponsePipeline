import os
import re

import pandas as pd
import numpy as np
import sqlalchemy

from nltk.tokenize import word_tokenize


def load_data(path: str = "../data/DisasterResponse.db"):
    # read data from SQLite database into a dataframe
    engine = sqlalchemy.create_engine(os.path.join("sqlite:///", path))
    df = pd.read_sql_table("DisasterResponseTable", con=engine)

    # separate data into features and targets
    x = df["message"].copy()
    y = df.iloc[:, 4:]
    return x, y


def tokenize(text: str):
    """
    Tokenizes a text string
    :param text: string to tokenize
    :return: list of tokens
    """
    words
    return clean_tokens


print(x)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
detected_urls = re.findall(url_regex, text)
