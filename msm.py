import argparse
import pandas as pd
import numpy as np
import re
import sys

from collections import OrderedDict
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from ordered_set import OrderedSet
from similarities import *


def same_shop(product_1, product_2, debug):

    if debug:
        print(product_1.get("shop"))
        print(product_2.get("shop"))

    return product_1.get("shop") == product_2.get("shop")


def same_brand(product_1, product_2, brands, debug):
    product_1_brand = "NA"
    product_2_brand = "NA"

    if product_1.get("featuresMap").get("Brand") is not None:
        product_1_brand = product_1.get("featuresMap").get("Brand").lower()

    if product_2.get("featuresMap").get("Brand") is not None:
        product_2_brand = product_2.get("featuresMap").get("Brand").lower()

    if product_1_brand == "NA":
        for key in brands:
            if re.search(rf'\b{key}\b', product_1.get("title").lower()):
                product_1_brand = key
                break
    
    if product_2_brand == "NA":
        for key in brands:
            if re.search(rf'\b{key}\b', product_2.get("title").lower()):
                product_2_brand = key
                break

    if debug:
        print(product_1_brand)
        print(product_2_brand)

    return product_1_brand == product_2_brand or product_1_brand == "NA" or product_2_brand == "NA"


def same_resolution(product_1, product_2, debug):
    product_1_reso = "NA"
    product_2_reso = "NA"

    regex = r'(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    featuresMap_product_1 = product_1.get("featuresMap")
    featuresMap_product_2 = product_2.get("featuresMap")

    for key in featuresMap_product_1:
        if "resolution" in key.lower():
            temp = re.search(regex, featuresMap_product_1.get(key))
            if temp is not None:
                product_1_reso = temp.group(0)
                break
 
    for key in featuresMap_product_2:
        if "resolution" in key.lower():
            temp = re.search(regex, featuresMap_product_2.get(key))
            if temp is not None:
                product_2_reso = temp.group(0)
                break

    if debug:
        print(product_1_reso)
        print(product_2_reso)

    return product_1_reso == product_2_reso or product_1_reso == "NA" or product_2_reso == "NA"


def extract_model_words(features, keys):
    key_words_regex = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    model_words = OrderedSet()

    for key in keys:
        if key in features: 
            matches = re.findall(key_words_regex, features.get(key, ""))
            model_words.update(matches)

    return model_words


def title_comp(title_1, title_2, alpha, beta, delta, approx):
    title_regex = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'

    name_cosine_sim = cosineSim(title_1, title_2)

    if name_cosine_sim > alpha:
        return 1

    model_words_1 = OrderedSet()
    model_words_2 = OrderedSet()

    model_words_1.update(x[0] for x in re.findall(title_regex, title_1))
    model_words_2.update(x[0] for x in re.findall(title_regex, title_2))

    similar_model_words = False

    for word_1 in model_words_1:
        non_numeric_1, numeric_1 = split_numeric(word_1)

        for word_2 in model_words_2:
            non_numeric_2, numeric_2 = split_numeric(word_2)

            approx_sim = norm_lv(non_numeric_1, non_numeric_2)

            if approx_sim > approx and numeric_1 != numeric_2:
                return -1
            elif approx_sim > approx and numeric_1 == numeric_2:
                similar_model_words = True
    
    final_name_sim = beta * name_cosine_sim + (1-beta) * avg_lv_sim(model_words_1=model_words_1, model_words_2=model_words_2, mw=False)

    if similar_model_words:
        final_name_sim = delta * avg_lv_sim(model_words_1=model_words_1, model_words_2=model_words_2, mw=True) + (1 - delta) * final_name_sim

    return final_name_sim
    


def clustering(dissimilarity_matrix, threshold):
    clustered= AgglomerativeClustering(metric="precomputed", linkage="complete", distance_threshold=threshold, n_clusters=None)
    clustered.fit(dissimilarity_matrix)

    return clustered


def main(args):
    dissimilarity: pd.DataFrame = args.candidate_pairs.copy()
    data: OrderedDict = args.data
    brands: OrderedDict = args.brand_list

    gamma: float = args.parameters.get("gamma")
    epsilon: float = args.parameters.get("epsilon")
    mu: float = args.parameters.get("mu")

    print("Begin MSM...\n")

    for i in range(len(dissimilarity)):
        for j in range(i+1, len(dissimilarity)):       
            row_name = dissimilarity.index[i]
            column_name = dissimilarity.columns[j]

            if dissimilarity.loc[row_name, column_name] == 0:
                product_1 = data.get(row_name)
                product_2 = data.get(column_name)

                if (same_shop(product_1, product_2, False) or (not same_brand(product_1, product_2,brands=brands, debug = False))
                     or (not same_resolution(product_1, product_2,debug = False))):
                    dissimilarity.loc[row_name, column_name] = 1
                    dissimilarity.loc[column_name, row_name] = 1
                    continue

                sim = 0
                mean_sim = 0

                m = 0 # Number of matches
                w = 0 # Weight of matches

                features_1 = product_1.get("featuresMap")
                features_2 = product_2.get("featuresMap")

                no_match_keys_1 = list(features_1.keys()).copy()
                no_match_keys_2 = list(features_2.keys()).copy()

                for key_1 in features_1.keys():
                    match = False

                    if not match:
                        for key_2 in no_match_keys_2:
                            key_sim = q_gram_similarity(string_1=key_1,string_2=key_2,q=3)
                            
                            if key_sim > gamma:
                                value_sim = q_gram_similarity(string_1=features_1.get(key_1), string_2=features_2.get(key_2), q=3)
                                sim = sim + key_sim * value_sim
                                m += 1
                                w = w + key_sim

                                match = True
                                
                                no_match_keys_1.remove(key_1)
                                no_match_keys_2.remove(key_2)
                                break

                if w > 0:
                    mean_sim = sim / w

                model_words_1 = extract_model_words(features=features_1, keys=no_match_keys_1)
                model_words_2 = extract_model_words(features=features_2, keys=no_match_keys_2)

                union_len = len(model_words_1.union(model_words_2))
                mw_percentage = 0 if union_len == 0 else len(model_words_1.intersection(model_words_2)) / union_len

                title_sim = title_comp(product_1.get("title"), product_2.get("title"), alpha=0.602, beta=0.0, delta=0.5, approx=0.5)

                if title_sim == -1:
                    theta_1 = m/min(len(features_1), len(features_2))
                    theta_2 = 1 - theta_1
                    h_sim = theta_1 * mean_sim + theta_2 * mw_percentage
                else:
                    theta_1 = (1 - mu) * m/min(len(features_1), len(features_2))
                    theta_2 = 1 - mu - theta_1
                    h_sim = theta_1 * mean_sim + theta_2 * mw_percentage + mu * title_sim
                
                dissimilarity.loc[row_name, column_name] = 1 - h_sim
                dissimilarity.loc[column_name, row_name] = 1 - h_sim

    np.fill_diagonal(dissimilarity.values, 0)

    print("Begin clustering...\n")
    return clustering(dissimilarity_matrix=dissimilarity, threshold=epsilon), dissimilarity


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",type=OrderedDict, help="The data that is used to train the model.")
    parser.add_argument("--parameters",type=dict, help="Parameters for this the methodology.")
    parser.add_argument("--candidate_pairs", type = pd.DataFrame, help="Matrix of candidate pairs (cleaned)")
    parser.add_argument("--brand_list", type=OrderedDict, help="Brand List")

    args = parser.parse_args()

    main(args)