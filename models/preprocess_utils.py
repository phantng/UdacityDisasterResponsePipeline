import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


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
