# MSMP++: Scalable Duplicate Detection

Source code for the Multi-Similarity Model with Preselection++ (MSMP++). This is a scalable duplicate detection model which utilises Minhashing, LSH and the MSM methods, and build upon the MSM, MSMP and MSMP+ model.

## Data

The data that is used within this GitHub is uploaded in the `TVs-all-merged.json` file. Note that the path needs to be adjusted in the `main.py` file such that it is passed to the main method of `data_clean.py`.

## Running
`data_clean.py`: cleans the data by removing superfluous words and symbols. Defines a splitting function for bootstrapping, guaranteeing both the train and the tes data contain product duplicates.

`binary_vectors.py`: converts each product in the given data to a binary vector using model words which are extracted from the dataset using regex equations.

`min_hash.py`: carries out the min-hashing (dimension reduction) using the binary vectors created from `binary_vectors.py`.

`lsh.py`: creates candidate pairs through local-sensitivity hashing, which hashes snippets of the product representations from the signature matrix of `min_hash.py` into buckets.

`msm.py`: implements the multi-component similarity method. Returns clusters of candidate pairs.

`evaluation.py`: evaluates the clusters returned by msm and returns various evaluation metrics. Also includes a function used for the graphing of results.

`main.py`: combines all the previous functions into a working model architecture. This can be run in one go.

`similarities.py`: defines various similarity functions which are used throughout the project.

