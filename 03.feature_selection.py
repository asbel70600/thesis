import math
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


def common_language_effect_size(sample1, sample2):
    # Calculate the means and standard deviations of each sample
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1), np.std(sample2)

    pooled_std = np.sqrt(
        (((len(sample1) - 1) * std1**2) + (len(sample2) - 1) * std2**2)
        / (len(sample1) + len(sample2) - 2)
    )

    if pooled_std == 0:
        pooled_std = sys.float_info.min

    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std

    # Calculate CLES using the normal cumulative distribution function (CDF)
    cles = norm.cdf(d / np.sqrt(2))
    return cles


def isNormal(data: NDArray) -> bool:
    if stats.normaltest(data)[1] > 0.05:
        return True
    else:
        return False


def evaluate(data: tuple[str, list[NDArray]]) -> tuple[str, float]:
    (feature_name, populations) = data
    normality = [isNormal(x) for x in populations]
    means = [np.mean(x) for x in populations]
    stds = [np.std(x) for x in populations]
    lens = [len(x) for x in populations]
    pvalues: NDArray[np.float64] = np.zeros((len(populations), len(populations)))

    # for current in range(len(populations)):
    #     if not normality[current]:
    #         continue
    #     for other in range(current + 1, len(populations)):
    #         if not normality[other]:
    #             continue
    #         cles = common_language_effect_size(populations[current], populations[other])
    #         pval = stats.ttest_ind_from_stats(
    #             means[current],
    #             stds[current],
    #             lens[current],
    #             means[other],
    #             stds[other],
    #             lens[other],
    #             equal_var=False,
    #         )[1]
    #         pvalues[current][other] = (1.0 / pval)*(cles/2.0)

    for current in range(len(populations)):
        for other in range(current + 1, len(populations)):
            cles = common_language_effect_size(populations[current], populations[other])
            pval = stats.mannwhitneyu(populations[current], populations[other]).pvalue
            pvalues[current][other] += (1/(1+math.e**-((1.0/pval))))*cles

    overallp = np.sum(pvalues.flatten())

    return (feature_name, overallp.astype(float))


if __name__ == "__main__":
    dataset_paths: list[Path] = []
    populations: list[tuple[str, DataFrame]]
    populations = []
    feature_names = []
    dataset_paths = []

    for i in Path("data").glob("*"):
        dataset_paths.append(i)

    for i in dataset_paths:
        if os.path.exists(i) and os.path.isfile(i) and os.path.getsize(i) != 0:
            data_raw = pd.read_csv(i)

            tmp = data_raw["class"].astype(dtype=str)[0]

            if type(tmp) == str:
                classification: str = tmp
            else:
                print(f"Oh god this isn't a string {tmp}")
                classification = ""

            data_noclass = data_raw.drop(columns=["class"]).astype(float)
            feature_names: list[str] = data_noclass.keys().tolist()

            populations.append((classification, data_noclass))

    iterations: list[tuple[str, list[NDArray]]] = []
    same_column_collection: list[NDArray] = []

    for feature in feature_names:
        same_column_collection = []
        for population in populations:
            same_column_collection.append(np.array(population[1][feature].tolist()))

        iterations.append((feature, same_column_collection))

    with Pool(12) as p:
        eso = p.map(evaluate, iterations)

    aquello = sorted(eso, key=lambda x: x[1])

    d = []
    for i in aquello[-10:]:
        d.append(i[0])

    d.insert(0, "class")
    for i in populations:
        df = i[1]
        classif = i[0]
        df.insert(0, "class", classif)
        if not os.path.exists("data"):
            os.mkdir("data")

        if not os.path.exists("data/filtered"):
            os.mkdir("data/filtered")

        df[d].to_csv(f"data/filtered/{classif}_filtered.csv", index=False)

# def main():
#     p = 0.05
#
#     pink = pd.read_csv("data/pink_headers.csv")
#     yellow = pd.read_csv("data/yellow_headers.csv")
#
#     pink_numeric = DataFrame(pink.loc[:, pink.columns != "class"]).astype(float)
#     pn = DataFrame(pink.where)
#     yellow_numeric = DataFrame(yellow.loc[:, yellow.columns != "class"]).astype(float)
#
#     (nyellow, ntnyellow) = get_normal_columns(pink_numeric)
#     (npink, ntnpink) = get_normal_columns(yellow_numeric)
#
#     difference_score = {x: 0.0 for x in pink_numeric.keys()}
#     ttest_score = {x: 0.0 for x in pink_numeric.keys()}
#     mw_score = {x: 0.0 for x in pink_numeric.keys()}
#
#     common_normal = nyellow & npink
#     common_notnormal = ntnyellow & ntnpink
#     not_coinc = (nyellow ^ npink) | (ntnyellow ^ ntnpink)
#
#     for i in common_normal:
#         meana = np.mean(pink_numeric[i])
#         meanb = np.mean(yellow_numeric[i])
#
#         stda = np.std(pink_numeric[i])
#         stdb = np.std(yellow_numeric[i])
#
#         lena = len(pink_numeric[i])
#         lenb = len(yellow_numeric[i])
#
#         # cohen = cohen_d(pink_numeric[i], yellow_numeric[i])
#         cles = common_language_effect_size(pink_numeric[i], yellow_numeric[i])
#
#         pval = stats.ttest_ind_from_stats(
#             meana, stda, lena, meanb, stdb, lenb, equal_var=False
#         )[1]
#
#         ttest_score[i] += (1 / pval) * cles
#
#     for i in common_notnormal:
#         cles = common_language_effect_size(pink_numeric[i], yellow_numeric[i])
#         pval = stats.mannwhitneyu(pink_numeric[i], yellow_numeric[i]).pvalue
#         mw_score[i] = (1 / pval) * cles
#
#     for i in not_coinc:
#         cles = common_language_effect_size(pink_numeric[i], yellow_numeric[i])
#         pval = stats.mannwhitneyu(pink_numeric[i], yellow_numeric[i]).pvalue
#         mw_score[i] = (1 / pval) * cles
#
#     final_score = {x: 0.0 for x in pink_numeric.keys()}
#
#     for i in final_score.keys():
#         final_score[i] += difference_score[i]
#         final_score[i] += ttest_score[i]
#         final_score[i] += mw_score[i]
#
#     keys = list(final_score.keys())
#     values = list(final_score.values())
#     indexes = np.argsort(values)
#     better = {keys[i]: values[i] for i in indexes}
#
#     top_10 = list(better.keys())[-10:]
#     top_10.append("class")
#
#     for i in pink_numeric.keys():
#         if i not in top_10:
#             pink_numeric = pink_numeric.drop(columns=[i])
#             pink = pink.drop(columns=[i])
#
#     for i in yellow_numeric.keys():
#         if i not in top_10:
#             yellow_numeric = yellow_numeric.drop(columns=[i])
#             yellow = yellow.drop(columns=[i])
#
#     pink = pd.concat([pink[['class']], pink_numeric], axis=1)
#     yellow = pd.concat([yellow[['class']], yellow_numeric], axis=1)
#
#     pink.to_csv("data/pink_headers_filtered.csv",index=False)
#     yellow.to_csv("data/yellow_headers_filtered.csv", index=False)
#
#     merged = pd.concat([pink, yellow], ignore_index=True)
#     shuffled = merged.sample(frac=1).reset_index(drop=True)
#
#     shuffled.to_csv("data/shuffled.csv",index=False)
#
#     print(top_10)
#
#     # print(ttest_score)
#     # print(sys.float_info.max)
#
#     for i in difference_score.keys():
#         if i in not_coinc:
#             difference_score[i] += 1 / p
#
#     # print(difference_score.values())


# arenormal=np.empty(int(len(data)/2))
#     arenotnormal=np.empty(int(len(data)/2))
#
#     for i in data.keys():
#         if i == "class":
#             continue
#
#         da = pink[i].astype(float)
#         db = yellow[i].astype(float)
#
#         normala=get_normal_columns(da)
#         normalb=get_normal_columns(db)
#
#
#         (s,p) = stats.normaltest(da,nan_policy='omit')
#         if p > 0.05:
#             print(i)
#             print(f"stats {s}")
# , ddof=1             print(f"pvalue {p}")


# def get_normal_columns(data: pd.DataFrame) -> tuple[set[str], set[str]]:
#     normal = set()
#     notnormal = set()
#
#     for i in data.keys():
#
#         if stats.normaltest(data[i].astype(float))[1] > 0.05:
#             normal.add(i)
#         else:
#             notnormal.add(i)
#
#     return (normal, notnormal)


# def cohen_d(sample1, sample2):
#     # Calculate the means of each sample
#     mean1, mean2 = np.mean(sample1), np.mean(sample2)
#
#     # Calculate the standard deviations of each sample
#     std1, std2 = np.std(sample1), np.std(sample2)
#
#     # Calculate the pooled standard deviation
#     pooled_std = np.sqrt(
#         (((len(sample1) - 1) * std1**2) + (len(sample2) - 1) * std2**2)
#         / (len(sample1) + len(sample2) - 2)
#     )
#
#     # Calculate Cohen's d
#     d = (mean1 - mean2) / pooled_std
#     return d
#
