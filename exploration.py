import os

import pandas as pd

PATH = os.path.join("datasets", "collection", "collection.tsv")
data = pd.read_csv(PATH, nrows=10, sep="\t")
print(data.head())