import argparse
import numpy as np 

from primePy import primes
from collections import OrderedDict

""" Creates a min-hashed signature matrix
:param binary_vectors: the set of binary vectors to be minhashed
:param reduction: the factor of reduction
"""
def min_hash(binary_vectors, reduction):

    num_vec = len(binary_vectors)
    len_vec = len(binary_vectors.get(list(binary_vectors.keys())[0]))

    n = int(round(reduction * len_vec / 500) * 500)

    sig_matrix = np.full((n,num_vec), np.inf)

    list_primes = primes.between(n+1, n+100000)
    index = np.random.randint(len(list_primes))
    p = list_primes[index]

    a = np.random.randint(0,p, size = n)
    b = np.random.randint(0,p, size = n)

    for i in range(len_vec):
        hashed_values = [(a[j] + b[j] * i) % p for j in range(n)]

        col_index = 0
        for key in binary_vectors:
            vector = binary_vectors.get(key)
            if (vector[i] == 1): 
                sig_matrix[:, col_index] = np.minimum(hashed_values, sig_matrix[:, col_index])
            col_index += 1

    return sig_matrix


def main(args):
    binary_vectors: OrderedDict = args.binary_vectors

    sig_matrix = min_hash(binary_vectors, 0.5)

    return sig_matrix


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--binary_vectors",type=OrderedDict, help="The dictionary of binarised vectors")
    args = parser.parse_args()

    main(args)