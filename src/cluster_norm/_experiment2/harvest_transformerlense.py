import os, sys, gc
from pathlib import Path

import torch as t
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
print("Using device:", device)

experiment2_path = "/home/wombat_share/laurito/cluster_norm/src/cluster_norm/_experiment2"

experiment2_path = "/home/wombat_share/laurito/cluster_norm/src/cluster_norm/_experiment2"

def free_mem(vars):
    for v in vars: del v
    gc.collect()
    t.cuda.empty_cache()



if __name__ == "__main__":
    model_name, layer, sentiment, variant = sys.argv[1:5]

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    layers = model.cfg.n_layers - 1
    d_model = model.cfg.d_model
    # tokenizer = AutoTokenizer.from_pretrained(models[model_name])

    templates = pd.read_json(f"{experiment2_path}/prompts_{sentiment}.jsonl", orient="records", lines=True)

    outpath = f"{experiment2_path}/transformer_lense/activations/{model_name}/layer{layer}"
    Path(outpath).mkdir(exist_ok=True, parents=True)
    activations = t.zeros(len(templates), d_model)

    hook_pt = f"blocks.{layer}.hook_resid_pre"

    for i in range(len(templates)):
        prompt = templates.at[i, f"variant{variant}"]
        tks = model.to_tokens(prompt)
        
        with t.no_grad():
            logits, cache = model.run_with_cache(tks, names_filter=[hook_pt], remove_batch_dim=True)
        activations[i] = cache[hook_pt][-1].cpu()
        
        free_mem([tks, logits, cache])
        
    t.save(activations, f"{outpath}/variant{variant}_{sentiment}.pt")
    free_mem([activations])
