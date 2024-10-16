import pandas as pd
import numpy as np


def main():
    pink = pd.read_csv("data_pink.csv")
    yellow = pd.read_csv("data_yellow.csv")

    print(pink.columns.take([0]))

if __name__ == "__main__":
    main()
