# %%
from datasets import load_dataset
import numpy as np
import pandas as pd
import random


# %%
dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
df = pd.DataFrame(dataset)

# %%
labels = ["company", 
          "educational institution",
          "artist",
          "athlete",
          "office holder",
          "mean of transportation",
          "building",
          "natural place",
          "village",
          "animal",
          "plant",
          "album",
          "film",
          "written work"]
labels = {i: l for i, l in enumerate(labels)}

# %%
# sample is 1000 examples w/ correct label company
# followed by 1000 examples w/ correct label not-company
sample_company = df[df.label == 0].sample(n=1000, random_state=123456)
sample_noncompany = df[df.label != 0].sample(n=1000, random_state=123456)
sample = pd.concat([sample_company, sample_noncompany])

# note down the correct labels, randomly sample an incorrect label
sample["correct"] = sample["label"].apply(lambda i: labels[i])
sample["incorrect"] = sample["label"].apply(
    lambda i: labels[random.sample([j for j in range(14) if j != i], 1)[0]]
    )
sample.drop(columns=["label", "title"], inplace=True)

# %%
suffix = """\
Example 4: {CONTENT}
Choice 1: {LABEL0}
Choice 2: {LABEL1}
Alice says "Choice """

def get_suffix(row):
    '''
    Randomly assign the correct/incorrect labels in the prompt.
    '''
    p = np.random.uniform()
    label0 = row.correct if p > 0.5 else row.incorrect
    label1 = row.correct if label0 == row.incorrect else row.incorrect
    return suffix.format(CONTENT=row.content,
                         LABEL0=label0,
                         LABEL1=label1)

sample["template_suffix"] = sample.apply(get_suffix, axis=1)

def get_option(row):
    '''
    Note down the choice number corresponding to the correct choice.
    '''
    ix = row["template_suffix"].index(f": {row['correct']}") - 1
    return int(row["template_suffix"][ix])
sample["choice_label"] = sample.apply(get_option, axis=1)

# %%
sample.to_json(f"prompts.jsonl", orient="records", lines=True)