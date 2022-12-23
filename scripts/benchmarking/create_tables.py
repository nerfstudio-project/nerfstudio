"""
Code to create the tables.
"""

import pandas as pd

# from nerfstudio.scripts.benchmarking.benchmark_nerfacto import table_rows

# the csv filenames
csv_filename = "wandb-psnr.csv"

# code to read the csv file

df = pd.read_csv(csv_filename)

column_titles = list(df.columns.values)
last_row = list(df.values[-1])

print(len(column_titles))
print(len(last_row))

from collections import defaultdict
results = defaultdict(dict)
for column_title, last_row_value in zip(column_titles, last_row):
    if column_title == "Step" or "__MAX" in column_title or "__MIN" in column_title:
        continue
    name = column_title.split(" - ")[0]
    capture_name, method = name.split("_")
    metric = float(last_row_value)
    # format to have 2 decimal places
    metric = f"{metric:.2f}"
    results[capture_name][method] = metric

import pprint
pprint.pprint(results)
print(len(results))
