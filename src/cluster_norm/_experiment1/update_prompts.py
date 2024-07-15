# imports all json files in "./prompt_datasets/ with pathlib:

import json
from pathlib import Path

import pandas as pd

from cluster_norm.utils.utils import extract_last_word

# Path to the prompt datasets
prompt_datasets = Path("./prompt_datasets/")
prompt_files = prompt_datasets.glob("*.jsonl")


for prompt_file in prompt_files:
    prompts = pd.read_json(prompt_file, orient="records", lines=True)
    # add distraction key to each json object in the prompt 
    prompts["distraction"] = prompts["template_pos_bs"].apply(extract_last_word)
    # Save the modified data to a new JSONL file
    prompts.to_json(prompt_file, orient="records", lines=True)
        

