from numpy.ma import shape
from scipy.stats import norm
import sys
from numpy._typing import NDArray
import pandas as pd
import numpy as np
from pandas.compat import os
from pandas.core.api import DataFrame
import scipy.stats as stats
from pathlib import Path
from multiprocessing import Pool

prelude = "."
input_path = f"{prelude}/data/filtered"
output_path = f"{prelude}/data/train/data.csv"

if not os.path.exists("data/train"):
    os.mkdir("data/train")

if not os.path.exists("data"):
    os.mkdir("data")

dataset_paths:list[Path]=[]
every_one:DataFrame = DataFrame()

for i in Path(input_path).glob("*"):
    dataset_paths.append(i)

for i in dataset_paths:
    if os.path.exists(i) and os.path.isfile(i) and os.path.getsize(i) != 0:
        data_raw = pd.read_csv(i)
        every_one = pd.concat([data_raw,every_one],ignore_index= True)

shuffled_df = every_one.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_df = shuffled_df.dropna()
shuffled_df.to_csv(output_path,index=False)
