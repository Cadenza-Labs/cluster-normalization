import argparse
import gc
import os
import pickle
from pathlib import Path

import pandas as pd
import torch as t
from tqdm import trange
from transformer_lens import HookedTransformer

# Set up environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = t.device("cuda:5" if t.cuda.is_available() else "cpu")


def harvest_activations(model, file_path, output_dir):
    """
    Process templates to extract and save activations.
    Args:
    file_path (Path): Path to the JSONL file containing templates.
    output_dir (Path): Directory to save the activation tensors.
    """
    # Load templates
    try:
        templates = pd.read_json(str(file_path), orient="records", lines=True)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return
    
    model_layers = model.cfg.n_layers - 1
    layers = [model_layers // 4, model_layers // 2, 3 * model_layers // 4, model.cfg.n_layers - 1]

    for layer in layers:
        hook_pt = f"blocks.{layer}.hook_resid_pre"
        output = output_dir / str(layer)

        output.mkdir(parents=True, exist_ok=True)

        # Process each template
        for c in ["pos", "neg"]:
            for test in ["", "_bs"]:
                activation_file = output / f"{c}{test}.pt"
                if activation_file.exists():
                    # avoid recreating activations if they already exist
                    print(f"Skip: activation_file already exists at {activation_file}")
                else:
                    column = f"template_{c}{test}"
                    activations = t.zeros(len(templates), model.cfg.d_model) 
                    for i in trange(len(templates), desc=f"Processing {column}"):
                        prompt = templates.at[i, column]
                        tks = model.to_tokens(prompt)
                        with t.no_grad():
                            logits, cache = model.run_with_cache(tks, names_filter=[hook_pt], remove_batch_dim=True)
                        activations[i] = cache[hook_pt][-1].cpu()

                        del tks, logits, cache
                        gc.collect()
                        t.cuda.empty_cache()

                    # Save the activations
                    t.save(activations, str(activation_file))
                    del activations
                    gc.collect()
                    print(f"Saved activations to {activation_file}")

# This labels are specific to imdb "positive" and "negative
def harvest_llm_answers(model, prompts, logits_dir, pseudo_labels):
    prompts = pd.read_json(str(prompt_path), orient="records", lines=True)
    tk_pos = model.to_tokens("positive").squeeze(0)[-1]
    tk_neg = model.to_tokens("negative").squeeze(0)[-1]
    
    model_layers = model.cfg.n_layers - 1
    layers = [model_layers // 4, model_layers // 2, 3 * model_layers // 4, model.cfg.n_layers - 1]
    
    for layer in layers:
        print("## layer", layer)
        output_dir = logits_dir / str(layer)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pseudo_label in pseudo_labels:
            for biased in [False, True]:
                filename = f"{output_dir}/{pseudo_label}_biased.pickle" if biased else f"{output_dir}/{pseudo_label}_unbiased.pickle"
                if not os.path.exists(filename):                    
                    answers = []
                    print("Processing", filename)
                    for i in trange(len(prompts), desc=filename.replace(str(output_dir) + "/", "")):
                        base_colname = "template_pos" if pseudo_label == pseudo_labels[0] else "template_neg"
                        suffix = "_bs" if biased else ""
                        colname = f"{base_colname}{suffix}"
                        prompt = prompts.at[i, colname]
                        tks = model.to_tokens(prompt)
                        with t.no_grad(): logits = model(tks, return_type="logits")
                        logits = logits[:, -1, [tk_pos, tk_neg]]
                        zero_shot = pseudo_labels[logits.squeeze(0).argmax().item()] 
                        answers.append(zero_shot)
                        del tks, logits
                        gc.collect()
                        t.cuda.empty_cache()
                        
                    outfile = open(filename, "wb")
                    pickle.dump(answers, outfile); outfile.close() 
                    print("LLM answers saved to", filename)
                else:
                    print("LLM answers already exist at", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest activations and logits.")
    parser.add_argument("--model", type=str, default="mistral-7b", help="model to use (default: mistral-7b)")
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use (default: imdb)")
    args = parser.parse_args()
    
    directory_path = Path("prompt_datasets") / args.dataset
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    print(f"{args.model} has {model.cfg.n_layers} layers")
    
    files = directory_path.glob("*.jsonl")
    # sort by lowest number to hights
    files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
    for prompt_path in files:
        # Extract 'k' from the filename assuming the format is like 'prompts_32.jsonl'
        num_random_words = prompt_path.stem.split("_")[1]
        logits_dir = Path("logits") / args.model / args.dataset / num_random_words
        activation_dir = Path("activations") / args.model / args.dataset / num_random_words
        
        # if args.dataset == "imdb":
        #     pseudo_labels = ["positive", "negative"]
        # harvest_llm_answers(m, prompt_path, logits_dir, pseudo_labels)
        harvest_activations(model, prompt_path, activation_dir)
