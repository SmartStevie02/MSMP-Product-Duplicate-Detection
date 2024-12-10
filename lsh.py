import argparse
import pandas as pd
import numpy as np
import mmh3
import msm


from collections import defaultdict, OrderedDict
from functools import reduce

""" Implements Local-Sensitivity Hashing
:param sig_matrix: signature matrix M 
:param data: the data to be made into candidate pairs
:b: num bands the matrix
"""
def LSH(sig_matrix, data: OrderedDict,  b: int):
    candidate_pairs = defaultdict(list)

    n = sig_matrix.shape[0]
    col = sig_matrix.shape[1]

    r = n // b
    
    threshold = (1/b)**(1/r)

    print(f"Based on chosen value b = {b} and the column length n = {n}, the approximate similarity threshold is {threshold}.")

    keys = list(data.keys())

    for j in range(col):
        for band in range(b):
            start = band * r
            band = sig_matrix[start:start + r,j]

            band_bytes = band.tobytes()
            band_hash = mmh3.hash(band_bytes, seed=42)
            
            candidate_pairs[band_hash].append(keys[j])

    return candidate_pairs, threshold

 
def cleaning_pairs(candidate_pairs_raw: defaultdict, data: OrderedDict):
    products = list(data.keys())

    candidate_pairs_df = pd.DataFrame(np.ones((len(products), len(products))), columns = products, index = products)

    for name in candidate_pairs_raw:
        potential_pair = candidate_pairs_raw.get(name)
        size_pairs = len(potential_pair)

        if size_pairs > 1:
            for i in range(size_pairs):
                for j in range(i+1, size_pairs):
                    product_1 = potential_pair[i]
                    product_2 = potential_pair[j]

                    candidate_pairs_df.loc[product_1,product_2] = 0
                    candidate_pairs_df.loc[product_2,product_1] = 0

    return candidate_pairs_df


def main(args):
    data: OrderedDict = args.data
    sig_matrix: int = args.sig_matrix
    b: int = args.b

    candidate_pairs_raw, threshold = LSH(sig_matrix=sig_matrix, data=data, b=b)
    candidate_pairs = cleaning_pairs(candidate_pairs_raw=candidate_pairs_raw, data=data)

    return candidate_pairs, threshold


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--binary_vectors",type=OrderedDict, help="The dictionary of binarised vectors")
    parser.add_argument("--sig_matrix", help="The signature matrix obtained during MinHashing")
    parser.add_argument("--b", type=int, help="The number of bands to use in LSH")

    args = parser.parse_args()

    main(args)


