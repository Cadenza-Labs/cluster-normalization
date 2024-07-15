# loade dataset from args.dataset in datasets folder:
import argparse
import json
from pathlib import Path

import pandas as pd


def process_imdb(dataset):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate processed dataset with templates.")
    parser.add_argument("--dataset", type=str, default="imdb", help="daataset to use (default: imdb)")
    args = parser.parse_args()

    path = Path(f"../datasets/raw/{args.dataset}.jsonl")
    with open(path, "r") as f:
        dataset = pd.read_json(path, orient="records", lines=True)
        
    if args.dataset == "imdb":
        dataset = process_imdb(dataset)

    # save pandas dataframe to jsonl file
    dataset.to_json(f"../datasets/{args.dataset}.jsonl", orient="records", lines=True)