import re
import argparse
import numpy as np


from collections import defaultdict, OrderedDict
from ordered_set import OrderedSet


def create_model_words(data, brands):
    title_regex = r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)'
    key_words_regex = r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)'

    model_words = OrderedSet()

    model_words.update(brands)

    for key in data:
        product = data.get(key)
        model_words.update(x[0] for x in re.findall(title_regex, product.get("title")))

        features = product.get("featuresMap")
        for key in features:
            model_words.update(re.findall(key_words_regex, features.get(key)))

    return(model_words)


def create_binary_vectors(model_words, data):
    print("Begin binary vector creation...\n")

    binary_vectors = OrderedDict()
    num_obs = len(data)

    tracking = 0
    for key in data:
        indiv_vector = np.zeros(len(model_words))
        product = data.get(key)

        index = 0 
        for word in model_words:
            if (word in product.get("title") or 
                any(word in value for value in product.get("featuresMap").values())):
                indiv_vector[index] = 1
            index += 1

        binary_vectors[key] = indiv_vector

        if tracking % 100 == 0:
            print(f"Creating binary vector {tracking} out of {num_obs}")

        tracking += 1

    print("\nCreation binary vectors complete.\n")

    return(binary_vectors)


def main(args):
    data: defaultdict = args.data
    brand_list: defaultdict = args.brand_list

    model_words = create_model_words(data,brand_list)
    binary_vectors = create_binary_vectors(model_words,data)

    return binary_vectors


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",type=OrderedDict, help="The dictionary of the data to be binarised.")
    parser.add_argument("--brand_list", type=OrderedDict, help="Brand List")
    args = parser.parse_args()

    main(args)
