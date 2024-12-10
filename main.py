import data_clean
import binary_vectors
import min_hash
import lsh
import msm
import evaluation

import json
import argparse
import numpy as np
import pandas as pd

from argparse import Namespace
from collections import defaultdict
from sympy import divisors
from evaluation import plot_metrics

def main(args):

    path: str = args.path
    path_res: str = args.path_res
    bootstraps: int = args.bootstraps

    file = open(path)
    data = json.load(file)

    results = {}

    for i in range(bootstraps):
        print(f"Running bootstrap {i+1} out of {bootstraps}\n")

        inputs = Namespace(data = data)
        train, test, brands_train, brands_test = data_clean.main(inputs)   

        inputs = Namespace(data = train, brand_list = brands_train)
        binary_rep_tune = binary_vectors.main(inputs)

        inputs = Namespace(binary_vectors = binary_rep_tune)
        sig_matrix_tune= min_hash.main(inputs)

        n = sig_matrix_tune.shape[0]
        factors = list(divisors(n)) 

        b_optimise = factors[len(factors) // 2]
        
        for factor in factors:
            r = n // factor
            threshold = (1/factor)**(1/r)
            
            if threshold <= 0.4 and threshold >= 0.15:
                b_optimise = factor
                break

        inputs = Namespace(data = train, sig_matrix = sig_matrix_tune, b = b_optimise)
        candidate_pairs_tune,_ = lsh.main(inputs)

        best_params = {}
        best_f1 = 0

        progress = 1

        gammas = [0.7, 0.75, 0.8]
        epsilons = [0.4, 0.5, 0.6]
        mus = [0.5, 0.65, 0.8]

        num_iter_grid = len(gammas) * len(epsilons) * len(mus)

        for gamma in gammas:
            for epsilon in epsilons:
                for mu in mus:
                    print(f"Grid Search Iteration: {progress} out of {num_iter_grid}")

                    params = {"gamma": gamma, "epsilon": epsilon, "mu": mu}
                    inputs = Namespace(data = train, parameters = params, candidate_pairs = candidate_pairs_tune, brand_list = brands_train)
                    clustered_tune, dissimilarity_tune = msm.main(inputs)

                    inputs = Namespace(data=train, dissimilarity = dissimilarity_tune, candidate_pairs = candidate_pairs_tune, clusters = clustered_tune)
                    f1 = np.array(evaluation.main(inputs))[0]

                    if f1 > best_f1:
                        best_params = params.copy()
                        best_f1 = f1
                    
                    progress += 1

        print("The best parameters are:")
        print(best_params)

        inputs = Namespace(data = test, brand_list = brands_test)
        binary_rep = binary_vectors.main(inputs)

        inputs = Namespace(binary_vectors = binary_rep)
        sig_matrix = min_hash.main(inputs)

        n = sig_matrix.shape[0]
        factors = list(divisors(n))

        columns = ["Threshold", "F1", "F1*", "PC", "PQ", "Number Comparisons", "Fraction Comparisons"]
        store_df = pd.DataFrame(np.zeros((len(factors), len(columns))), columns = columns, index = factors)

        for factor in factors:

            inputs = Namespace(data = test, sig_matrix = sig_matrix, b = factor)
            candidate_pairs, threshold = lsh.main(inputs)

            inputs = Namespace(data = test, parameters = best_params, candidate_pairs = candidate_pairs, brand_list = brands_test)
            clustered, dissimilarity = msm.main(inputs)

            inputs = Namespace(data=test, dissimilarity = dissimilarity, candidate_pairs = candidate_pairs, clusters = clustered)
            metrics = np.array(evaluation.main(inputs))
            
            full_store = np.append([threshold], metrics)

            print(full_store)
            store_df.loc[factor,] = full_store

        results[i] = store_df.copy()

    cumulative_df = pd.DataFrame(np.zeros((len(factors), len(columns))), columns = columns, index = factors)

    for i in results:
        cumulative_df += results.get(i)
    
    average_df = cumulative_df / bootstraps

    print("The final results are:")
    print(average_df)

    plot_metrics(average_df,"Fraction Comparisons","F1",path_res)
    plot_metrics(average_df,"Fraction Comparisons","F1*",path_res)
    plot_metrics(average_df,"Fraction Comparisons","PC",path_res)

    to_drop = average_df.copy() # Removes observations for Fraction Comp = 0 
    to_drop.drop(to_drop.loc[to_drop['Fraction Comparisons'] == 0].index, inplace=True)

    plot_metrics(to_drop,"Fraction Comparisons","PQ",path_res)


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", default="C:/Users/livin/OneDrive/University/Master/Block 2/Computer Science for BA/Assignment/Data/TVs-all-merged.json", 
                        type=str, help="The path to the data to be cleaned.")
    parser.add_argument("--path_res", default="C:/Users/livin/OneDrive/University/Master/Block 2/Computer Science for BA/Assignment/Results/", 
                        type=str, help="The path where the graphs are to be saved.")
    parser.add_argument("--bootstraps", default=5, type=int, help="The number of bootstraps you wish to do.")
    args = parser.parse_args()

    main(args)