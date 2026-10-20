#!/usr/bin/env python3
"""
1. Filter cluster-result files down to representative IDs.
2. Merge those IDs with extra info and split into train/benchmark sets.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Anchor all paths to this script's own location, so it works no matter
# what directory you run it from.
BASE_DIR = Path(__file__).resolve().parent

N_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# (label, ids_file, cluster_file, cluster_out_file, train_file, bench_file, info_file)
DATASETS = [
    (
        "positive",
        BASE_DIR / "positive_cluster" / "rep_positive.ids",
        BASE_DIR / "positive_cluster" / "cluster-results_cluster.tsv",
        BASE_DIR / "positive_cluster" / "pos_cluster_results.tsv",
        BASE_DIR / "cross_validation" / "pos_train.tsv",
        BASE_DIR / "cross_validation" / "pos_bench.tsv",
        BASE_DIR.parent / "01_data_collection" / "positives" / "positive.tsv",
    ),
    (
        "negative",
        BASE_DIR / "negative_cluster" / "rep_negative.ids",
        BASE_DIR / "negative_cluster" / "cluster-results_cluster.tsv",
        BASE_DIR / "negative_cluster" / "neg_cluster_results.tsv",
        BASE_DIR / "cross_validation" / "neg_train.tsv",
        BASE_DIR / "cross_validation" / "neg_bench.tsv",
        BASE_DIR.parent / "01_data_collection" / "negatives" / "negative.tsv",
    ),
]


def filter_by_ids(ids_file, cluster_file, out_file):
    """Write every line of cluster_file that contains one of the IDs."""
    ids = {line.strip() for line in open(ids_file) if line.strip()}

    with open(cluster_file) as cluster, open(out_file, "w") as out:
        for line in cluster:
            if any(seq_id in line for seq_id in ids):
                out.write(line.rstrip() + "\n")


def split_dataset(ids_file, train_file, bench_file, label, info_file):
    """Merge IDs with extra info, then split into train/benchmark sets."""
    ids = pd.read_csv(ids_file, sep="\t", header=None, names=["seq_id"])
    ids["seq_id"] = ids["seq_id"].str.strip()
    ids["class"] = label

    info = pd.read_csv(info_file, sep="\t").rename(columns={"Accession": "seq_id"})
    info["seq_id"] = info["seq_id"].str.strip()

    df = ids.merge(info, on="seq_id", how="inner")

    train, bench = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train = train.reset_index(drop=True)
    train["fold"] = [(i % N_FOLDS) + 1 for i in range(len(train))]

    train.to_csv(train_file, sep="\t", index=False)
    bench.to_csv(bench_file, sep="\t", index=False)

    print(f"  Training set:   {len(train)} sequences -> {train_file}")
    print(f"  Benchmark set:  {len(bench)} sequences -> {bench_file}")


if __name__ == "__main__":
    for label, ids_file, cluster_file, cluster_out, train_file, bench_file, info_file in DATASETS:
        print(f"Processing {label} dataset...")

        filter_by_ids(ids_file, cluster_file, cluster_out)
        print(f"  Filtered clusters -> {cluster_out}")

        split_dataset(ids_file, train_file, bench_file, label, info_file)
        print()