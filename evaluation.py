import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import OrderedDict, defaultdict
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations

def find_predictions(clusters):
    predictions = set()
    labels = clusters.labels_

    for i in range(clusters.n_clusters_):
        contained_in_i = np.where(labels == i)[0]
        
        if(len(contained_in_i) > 1):
            predictions.update(combinations(contained_in_i, 2))

    return(predictions)


def check_duplicates(data, dissimilarity, candidate_pairs, clusters):
    duplicates = set()
    predictions_lsh = set()

    duplicates_initial = defaultdict(list)
    
    product_names = dissimilarity.columns

    predictions = find_predictions(clusters)
    
    for index in range(len(product_names)):
        model_id = data.get(product_names[index]).get("modelID")
        duplicates_initial[model_id].append(index)

        for index_2 in range(index + 1, len(product_names)):
            if candidate_pairs.iloc[index, index_2] == 0:
                predictions_lsh.add((index, index_2))

    for value in duplicates_initial.values():
        if len(value) >= 2:
            duplicates.update(combinations(value, 2))
    
    TP_set = predictions.intersection(duplicates)
    FP_set = predictions.difference(duplicates)
    FN_set = duplicates.difference(predictions)

    DF_set = predictions_lsh.intersection(duplicates)

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    DF = len(DF_set)

    precision = TP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0

    np.fill_diagonal(candidate_pairs.values, 1)

    n_comp = ((candidate_pairs == 0).sum().sum()) / 2
    n_possible_comp = candidate_pairs.shape[0] * (candidate_pairs.shape[0] - 1) / 2

    frac_comp = n_comp / n_possible_comp
   
    PQ = DF/n_comp if n_comp != 0 else 0
    PC = DF/(len(duplicates))

    F_1 = 2 * (precision * recall)/(precision + recall) if (precision+recall) != 0 else 0
    F_1_star = (2 * PQ * PC)/(PQ+PC) if PQ+PC != 0 else 0

    return F_1, F_1_star, PC, PQ, n_comp, frac_comp


def plot_metrics(dataframe, metric1, metric2, path):
    os.makedirs(path, exist_ok=True)

    grouped_df = dataframe.groupby(metric1, as_index=False).agg({metric2: 'max'})

    horizontal = grouped_df[metric1]
    vertical = grouped_df[metric2]

    plt.figure(figsize=(8, 6))
    plt.plot(horizontal, vertical, marker='o', linestyle='-')

    title = f"{metric1} vs. {metric2}"

    plt.title(title)

    if "*" in title: 
        title = title.replace("*", "star")

    plt.xlabel(metric1)
    plt.ylabel(metric2)

    plt.grid(True)

    file_path = os.path.join(path, f"{title}.png")
    plt.savefig(file_path)

    plt.show()

def main(args):
    data: OrderedDict = args.data
    dissimilarity: pd.DataFrame = args.dissimilarity.copy()
    candidate_pairs: pd.DataFrame = args.candidate_pairs.copy()
    clusters: AgglomerativeClustering = args.clusters

    return check_duplicates(data=data, dissimilarity=dissimilarity, candidate_pairs=candidate_pairs, clusters=clusters)
    

if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",type=OrderedDict, help="The data that is used to train the model.")
    parser.add_argument("--dissimilarity", type = pd.DataFrame, help="The dissimilarity matrix used for clustering.")
    parser.add_argument("--candidate_pairs", type=pd.DataFrame, help="The matrix of candidate pairs.")
    parser.add_argument("--clusters",type=AgglomerativeClustering, help="The clusters found by the clustering algorithm.")
    
    args = parser.parse_args()

    main(args)