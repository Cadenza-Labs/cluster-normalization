import os, sys, gc
from pathlib import Path

import torch as t
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda:5" if t.cuda.is_available() else "cpu")

experiment2_path = "/home/wombat_share/laurito/cluster_norm/src/cluster_norm/_experiment2"

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()


# layers = [7, 15, 23, 31]
# d_model = 4096 if not "phi" in models else 2560
model_name, layer, sentiment, variant = sys.argv[1:5]

# load model and data
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    # cache_dir = llm_cache
)
model.eval()

layers = model.config.num_hidden_layers - 1
d_model = model.config.hidden_size
tokenizer = AutoTokenizer.from_pretrained(model_name)

templates = pd.read_json(f"{experiment2_path}/prompts_{sentiment}.jsonl", orient="records", lines=True)

outpath = f"{experiment2_path}/activations/{model_name}/layer{layer}"
Path(outpath).mkdir(exist_ok=True, parents=True)
activations = t.zeros(len(templates), d_model)
for i in range(len(templates)):
    prompt = templates.at[i, f"variant{variant}"]
    tks = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with t.no_grad(): out = model(tks, output_hidden_states=True)
    activations[i] = out["hidden_states"][int(layer)+1][0, -1, :].cpu()
    free_mem([tks, out])
t.save(activations, f"{outpath}/variant{variant}_{sentiment}.pt")
free_mem([activations])