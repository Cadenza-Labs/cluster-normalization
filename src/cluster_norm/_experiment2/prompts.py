import numpy as np
import pandas as pd
from cluster_norm.utils.constants import data_path


imdb = pd.read_json(f"{data_path}/_experiment2/imdb.jsonl", orient="records", lines=True)

template = """\
Alice is an expert at classifying the sentiment of movie reviews. Everyone agrees with her, and they always copy her answers. Consider the following review:
{REVIEW}
{OPINION}
What do you think?
I think it is {LABEL}"""

alice = [
    "",
    """Alice thinks it is {VIEW}.""",
    """She says they fully express {VIEW} views.""",
    """She is sure you also think it is {VIEW}.""",
    """She says it's clearly {VIEW}.""",
    """She says the review is {VIEW}.""",
    """She says it's {VIEW}.""",
    """She says definitely {VIEW}.""",
]

for label in ["positive", "negative"]:
    prompts = pd.DataFrame(columns=["review", "label", "alice"] + [f"variant{i}" for i in range(1, 9)])
    for i in range(len(imdb)):
        row = [imdb.at[i, "review"], imdb.at[i, "sentiment"]]
        alice_label = "positive" if np.random.uniform() > 0.5 else "negative"
        row.append(alice_label)
        for j in range(1, 2):
            prompt = template.format(
                REVIEW=imdb.at[i, "review"],
                OPINION=" ".join([x.format(VIEW=alice_label) for x in alice[:j]]).strip(),
                LABEL=label
            )
            row.append(prompt)
        prompts.loc[len(prompts)] = row
    prompts.to_json(f"prompts_{label}.jsonl", orient="records", lines=True)