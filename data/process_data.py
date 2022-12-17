# import necessary modules
import os
import sys

import pandas as pd
import numpy as np
import sqlalchemy


def load_data(messages_filepath: str = "messages.csv", categories_filepath: str = "categories.csv"):
    """
    Reads the data from .csv format into a Pandas dataframe
    :param messages_filepath: str path to messages.csv file
    :param categories_filepath:
    :return: pd.DataFrame of read data
    """
    # read categories data
    categories_df = pd.read_csv(categories_filepath)

    # read message data and merge on id with categories data
    message_df = pd.read_csv(messages_filepath)
    df = pd.merge(left=message_df,
                  right=categories_df,
                  on="id",
                  how="inner")
    return df


def clean_data(df):
    # explode the categories column categories dataframe
    exploded_cols = df["categories"].str.split(";", expand=True)
    df = pd.concat([df, exploded_cols], axis=1)
    df = df.drop("categories", axis=1)

    # parse the correct column names from the entries
    new_col_names = []
    for col in df.columns[4:]:
        new_col_names.append(df[col].iloc[0].split("-")[0])
    df.columns = df.columns.to_list()[:4] + new_col_names

    # binarize the target columns
    for col in df.columns[4:]:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: x.split("-")[1])

    # check for duplicates
    print("Number of duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates(subset=["message"], ignore_index=True)
    print("Number of duplicate rows after dropping:", df.duplicated().sum())

    return df


def save_data(df, database_filepath: str = "DisasterResponse.db"):
    # insert to database
    engine = sqlalchemy.create_engine(os.path.join("sqlite:///", database_filepath))
    df.to_sql("DisasterResponseTable", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print("Please provide the filepaths of the messages and categories "
              "datasets as the first and second argument respectively, as "
              "well as the filepath of the database to save the cleaned data "
              "to as the third argument. \n\nExample: python process_data.py "
              "disaster_messages.csv disaster_categories.csv "
              "DisasterResponse.db")


if __name__ == "__main__":
    main()
