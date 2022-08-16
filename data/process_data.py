# import necessary modules
import os

import pandas as pd
import numpy as np
import tabulate

# read and explode the categories column categories dataframe
categories_df = pd.read_csv("categories.csv")
exploded_cols = categories_df["categories"].str.split(";", expand=True)
categories_df = pd.concat([categories_df, exploded_cols], axis=1)
categories_df = categories_df.drop("categories", axis=1)

# parse the column names from the entries, and binarize the entries
new_col_names = []
for col in categories_df.columns[1:]:
    new_col_names.append(categories_df[col].iloc[0].split("-")[0])
categories_df.columns = categories_df.columns.to_list()[:1] + new_col_names
for col in categories_df.columns[1:]:
    categories_df.loc[:, col] = categories_df.loc[:, col].apply(lambda x: x.split("-")[1])

# read message data and merge on id with categories data
message_df = pd.read_csv("messages.csv")
df = pd.merge(left=message_df,
              right=categories_df,
              on="id",
              how="inner")

# check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())
print("Dropping duplicate rows")
df = df.drop_duplicates(ignore_index=True)
print("Number of duplicate rows after dropping:", df.duplicated().sum())
print(tabulate.tabulate(df.head(), header="keys"))