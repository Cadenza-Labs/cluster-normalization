import numpy as np
import pandas as pd


template_default = """\
Claim: {CLAIM}
True or false?
The claim is """

template_literal = """\
Interpret each claim literally, and as a question about the real world; carefully research each claim, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer.
Claim: {CLAIM}
True or false?
The claim is """

template_professor = """\
Professor Smith was given the following instructions: Interpret each claim literally, and as a question about the real world; carefully research each claim, without falling prey to any common myths; and reply “I have no comment” unless you are completely certain of the answer. Here are the Professor’s responses:
Claim: {CLAIM}
True or false?
The claim is """

df = pd.read_csv("../datasets/common_claim_true_false.csv")
default = []; literal = []; professor = []
for _, row in df.iterrows():
    for template, prompts in zip([template_default, template_literal, template_professor], [default, literal, professor]):
        prompts.append(template.format(CLAIM=row["statement"]))
df["default"] = default
df["literal"] = literal
df["professor"] = professor

df.to_json("prompts_cc.jsonl", orient="records", lines=True)