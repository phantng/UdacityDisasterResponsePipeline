import re

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score


def tokenize(text: str):  # , use_stemmer: bool = False, use_lemmatizer: bool = True):
    """
    Tokenizes a text string into a list of unique elements with punctuations stripped
    :param text: string to tokenize
    :return: list of tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # strip punctuation
    tokens = word_tokenize(text.lower())  # convert to lowercase and split text into unique words
    tokens = [w for w in tokens if w not in stopwords.words("english")]  # remove stopwords
    # if use_stemmer:
    #     tokens = [PorterStemmer().stem(w) for w in tokens]
    # if use_lemmatizer:
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]  # reduce word to lemma for keeping context

    return tokens


def compute_avg_columnwise_accuracy(y_true, y_pred):
    accuracy_results = []
    for idx, column in enumerate(y_true.columns):
        # compute individual column-wise accuracy
        accuracy = accuracy_score(y_true.values[:, column].values, y_pred[:, idx])
        accuracy_results.append(accuracy)
    return np.mean(accuracy_results)  # return simple average of column accuracies
