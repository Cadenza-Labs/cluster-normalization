import argparse
import json
from pathlib import Path

import pandas as pd

# Set up argument parsing
parser = argparse.ArgumentParser(description="Divide data into groups and apply random words.")
parser.add_argument("--dataset", type=str, default="imdb", help="daataset to use (default: imdb)")
parser.add_argument("--k", type=int, default=2, help="Number of groups to divide data into (default: 2)")
args = parser.parse_args()

# Retrieve the value of `k` from the arguments
k = args.k

file = Path(f"../../../datasets/{args.dataset}.jsonl")
# check if exists
if not file.exists():
    exit(f"File {file} does not exist.")

data = pd.read_json(file, orient="records", lines=True)

if args.dataset == "imdb":
    gt_labels = list(set(data["sentiment"]))
    
data["template_pos"] = data["template"].apply(lambda x: f"{x} {gt_labels[1]}")
data["template_neg"] = data["template"].apply(lambda x: f"{x} {gt_labels[0]}")

data["template_pos_bs"] = ""
data["template_neg_bs"] = ""

# Load random words from a JSON file
with open("random_words.json", "r") as file:
    random_words = json.load(file)["words"]

# Ensure enough words are available for the desired number of groups
if len(random_words) < k:
    raise ValueError(f"Insufficient words in the file to divide into {k} groups.")

# Select the first `k` words for grouping
group_words = {i: random_words[i] for i in range(k)}

# Randomly divide indices into `k` groups
groups = data.index.to_series().sample(frac=1).groupby(lambda x: x % k).indices

# Apply the words based on the assigned group
for group_num, indices in groups.items():
    word = group_words[group_num]
    for col in ["template_pos", "template_neg"]:
        data.loc[indices, f"{col}_bs"] = data.loc[indices, col].apply(lambda x: f"{x}. {word}")
        data.loc[indices, "distraction"] = word
        
prompt_folder = Path(f"./prompt_datasets/{args.dataset}")
prompt_folder.mkdir(parents=True, exist_ok=True)
data.to_json(f"{prompt_folder}/prompts_{k}.jsonl", orient="records", lines=True)
