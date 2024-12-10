import argparse
import re

from collections import OrderedDict
from sklearn.model_selection import train_test_split

def clean_data(data):
    cleaned_data_duplicates = OrderedDict()
    cleaned_data_non_duplicates = OrderedDict()

    clean_map_mw  = {
        "inch ": ["Inch","inches","\"", "\'", "”", "-inch", " inch", "inch"],
        "hz ": ["Hertz","hertz","Hz", "HZ", " hz", "-hz", "hz"]
    }

    clean_map_title = clean_map_mw
    clean_map_title[" "] = ["and", "or", "-", ",", "/", "&", "refurbished", "diagonal","diag.","best buy", "thenerds.net", "newegg.com"]

    for name in data:
        product = data.get(name)

        index = 0

        for product_per_shop in product:
            product_per_shop["title"] = replace(clean_map=clean_map_title, string=product_per_shop.get("title").lower())

            for feature, value in product_per_shop.get("featuresMap").items():
                product_per_shop["featuresMap"][feature] = replace(clean_map=clean_map_mw, string=value.lower())

            index += 1
    
        if index > 1:
            cleaned_data_duplicates[name] = product
        else:
            cleaned_data_non_duplicates[name] = product

    return cleaned_data_duplicates, cleaned_data_non_duplicates


def replace(clean_map, string):
    regex_clean = {
        re.compile(rf'{permutation}', re.IGNORECASE): replacement
        for replacement, permutations in clean_map.items()
        for permutation in permutations
    }

    for permutation, replacement in regex_clean.items():
        string = permutation.sub(replacement, string)

    return string


def test_train_split(cleaned_data_duplicates, cleaned_data_non_duplicates):
    duplicates_keys = list(cleaned_data_duplicates.keys())
    non_duplicates_keys = list(cleaned_data_non_duplicates.keys())
    
    duplicates_train_keys, duplicates_test_keys = train_test_split(duplicates_keys, train_size=0.67)
    non_duplicates_train_keys, non_duplicates_test_keys = train_test_split(non_duplicates_keys, train_size=0.67)

    duplicates_train = {key: cleaned_data_duplicates[key] for key in duplicates_train_keys}
    duplicates_test = {key: cleaned_data_duplicates[key] for key in duplicates_test_keys}
    non_duplicates_train = {key: cleaned_data_non_duplicates[key] for key in non_duplicates_train_keys}
    non_duplicates_test = {key: cleaned_data_non_duplicates[key] for key in non_duplicates_test_keys}

    train = restructure(duplicates_train)
    train.update(restructure(non_duplicates_train))

    test = restructure(duplicates_test)
    test.update(restructure(non_duplicates_test))

    return train, test


def restructure(data):
    new_data = OrderedDict()

    for name in data:
        product = data.get(name)

        for product_per_shop in product:
            shop = product_per_shop.get("shop")

            new_data[f"{name}_{shop}"] = product_per_shop

    return new_data


def create_brands(data):
    brands = OrderedDict()

    for key in data:
        brand_name = data.get(key).get("featuresMap").get("Brand")
        if brand_name is not None:
            brands[brand_name.lower()] = ""

    return brands


def main(args):
    data: dict = args.data

    print("Begin data cleaning...\n")
    cleaned_data_dup, cleaned_data_non= clean_data(data)
    train, test = test_train_split(cleaned_data_dup, cleaned_data_non)
    brands_train, brands_test = create_brands(train), create_brands(test)

    print("Data cleaning finished.\n")
    return train, test, brands_train, brands_test


if __name__ == "__main__":
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=dict, help="The data to be cleaned.")
    args = parser.parse_args()

    main(args)
