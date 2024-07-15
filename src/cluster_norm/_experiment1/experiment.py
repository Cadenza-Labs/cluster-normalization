import argparse
import os
import pickle
import sys

from transformer_lens import HookedTransformer

from cluster_norm._experiment1.update_prompts import extract_last_word
from cluster_norm.utils.CCS import CCS
from cluster_norm.utils.train import fit_ccs, fit_crc, fit_logreg
from cluster_norm.utils.utils import (create_accs_visualization,
                                      create_accs_visualization_2,
                                      evaluate_ccs, zero_shot)

sys.path.append("../utils/")

import numpy as np
import pandas as pd
import torch as t

runs_folder = "/home/wombat_share/laurito/cluster_norm/runs/experiment_1"


def fit_and_evaluate_methods(train_pos, test_pos, train_neg, test_neg, train_labels, test_labels, test_labels_bs, output_folder, neg_answers=None, pos_answers=None, n_probes=50):
    evals = {}
    
    # Zero Shot
    # acc_zs = zero_shot(pos_answers, neg_answers, test_labels)
    # evals["acc_zs"] = acc_zs
    
    # Logistic Regression
    lr = fit_logreg(train_pos, train_neg, train_labels)
    evals["acc_lg"] = lr.score(test_pos-test_neg, test_labels)
    evals["acc_lg_bs"] = lr.score(test_pos-test_neg, test_labels_bs)
    t.save(lr, f"{output_folder}/lr.pt")

    # CRC 
    for norm in ["burns", "cluster"]:
        crc = fit_crc(train_pos, train_neg, norm)
        t.save(crc, f"{output_folder}/crc_{norm}.pt")

        evals[f"acc_crc_{norm}"] = crc.evaluate(test_pos, test_neg, test_labels)
        evals[f"acc_crc_bs_{norm}"] = crc.evaluate(test_pos, test_neg, test_labels_bs)
    print("evals", evals)
    
    # # CRC 
    # crc = fit_crc(train_pos, train_neg, "burns")
    # acc_crc = crc.evaluate(test_pos, test_neg, test_labels)
    # evals["acc_crc_burns"] = acc_crc
    # t.save(crc, f"{output_folder}/crc_burns.pt")
    # print("evals", evals)

    # crc = fit_crc(train_pos, train_neg, "cluster")
    # acc_crc = crc.evaluate(test_pos, test_neg, test_labels)
    # evals["acc_crc_cluster"] = acc_crc
    # t.save(crc, f"{output_folder}/crc_cluster.pt")
    # print("evals", evals)
    
    # CCS
    viz_data = {}
    for norm in ["burns", "cluster"]:
        print(">> norm", norm)
        if args.load_from_cache and os.path.exists(f"{output_folder}/ccs_{norm}.pt"):
            print("Loading CCS from cache, skipping fitting")
            ccs = t.load(f"{output_folder}/ccs_{norm}.pt")
        else:
            _, ccs = fit_ccs(train_pos, train_neg, train_labels, normalize=norm, n_probes=n_probes)
            t.save(ccs, f"{output_folder}/ccs_{norm}.pt")
    
        sent_accs, bs_accs = evaluate_ccs(ccs, test_pos, test_neg, test_labels, test_labels_bs)
        evals[f"ccs_{norm} sent_accs (mean)"] = np.mean(sent_accs)
        evals[f"ccs_{norm} sent_accs (std)"] = np.std(sent_accs)
        evals[f"ccs_{norm} bs_accs (mean)"] = np.mean(bs_accs)
        evals[f"ccs_{norm} bs_accs (std)"] = np.std(bs_accs)
        print("evals", evals)
        
        viz_data[norm] = (sent_accs, bs_accs)
        
    # create_accs_visualization(viz_data, "ccs", output_folder=output_folder, n_probes=n_probes)

    return evals, viz_data
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment 1.")
    parser.add_argument("--model", type=str, default="mistral-7b", help="model to use (default: mistral-7b)")
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use (default: imdb)")
    parser.add_argument("--num_random_words", type=int, default=2, help="number of random words to use (default: 2)")
    parser.add_argument("--n_probes", type=int, default=50, help="number of probes to use (default: 50)")
    parser.add_argument("--load_from_cache", type=bool, default=True, help="load from cache (default: True)")
    parser.add_argument("--layers", nargs='+', type=int, help="List of layers to use", default=None)
    parser.add_argument("--split_ratio", type=float, default=0.8, help="split ratio (default: 0.8)")
    args = parser.parse_args()
    
    print(f"""model: {args.model}, dataset: {args.dataset}, num_random_words: {args.num_random_words}, 
          n_probes: {args.n_probes}, load_from_cache: {args.load_from_cache}, layers: {args.layers}, split_ratio: {args.split_ratio}""")

    activations_folder = f"activations/{args.model}/{args.dataset}/{args.num_random_words}"
    answers_root_folder = f"logits/{args.model}/{args.dataset}/{args.num_random_words}"
    
    prompt_file = f"./prompt_datasets/{args.dataset}/prompts_{args.num_random_words}.jsonl"

    prompts = pd.read_json(prompt_file, orient="records", lines=True)
    
    device = t.device("cuda:3" if t.cuda.is_available() else "cpu")
    if not args.layers:
        m = HookedTransformer.from_pretrained(args.model, device=device)
        m.eval()
        model_layers = m.cfg.n_layers - 1
        layers = [model_layers // 4, model_layers // 2, 3 * model_layers // 4, m.cfg.n_layers - 1]
    else:
        layers = args.layers
    for layer in layers:
        print("## layer", layer)
        pos = t.load(f"{activations_folder}/{layer}/pos.pt")
        neg = t.load(f"{activations_folder}/{layer}/neg.pt")
        pos_bs = t.load(f"{activations_folder}/{layer}/pos_bs.pt")
        neg_bs = t.load(f"{activations_folder}/{layer}/neg_bs.pt")
        
        # load pickles, so no torch can be used, since they are not tensors
        # pos_answers = pickle.load(open(f"{answers_root_folder}/{layer}/positive_unbiased.pickle", "rb")) 
        # neg_answers = pickle.load(open(f"{answers_root_folder}/{layer}/negative_unbiased.pickle", "rb"))
        # pos_answers_bs = pickle.load(open(f"{answers_root_folder}/{layer}/positive_biased.pickle", "rb"))
        # neg_answers_bs = pickle.load(open(f"{answers_root_folder}/{layer}/negative_biased.pickle", "rb"))
        
        if args.dataset == "imdb":
            labels = (prompts["sentiment"] == "positive").values
            
        last_word = extract_last_word(prompts["template_pos_bs"].iloc[0])
        labels_bs = prompts["template_pos_bs"].apply(lambda x: x.endswith(last_word)).values # change this

        perm = t.randperm(len(pos))
        pos, neg = pos[perm], neg[perm]
        pos_bs, neg_bs = pos_bs[perm], neg_bs[perm]
        labels, labels_bs = labels[perm], labels_bs[perm]
    
    
        viz_data = {}
        for current_run in ["non_bs", "bs"]:
            print("### current_run", current_run)
            output_folder = f"{runs_folder}/{args.model}/{args.dataset}/{args.num_random_words}/layer_{layer}/{current_run}"

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            if current_run == "bs":
                print("bias...")
                p, n = pos_bs, neg_bs
                # p_answer, n_answer = pos_answers_bs, neg_answers_bs
            else:
                print("no bias...")
                p, n = pos, neg
                # p_answer, n_answer = pos_answers, neg_answers
                
            split = int(args.split_ratio*len(p))
            train_pos, test_pos = p[:split], p[split:]
            train_neg, test_neg = n[:split], n[split:]
            train_labels, test_labels = labels[:split], labels[split:]
            train_labels_bs, test_labels_bs = labels_bs[:split], labels_bs[split:]
            
            evals, viz_method = fit_and_evaluate_methods(train_pos=train_pos, 
                                            test_pos=test_pos, 
                                            train_neg=train_neg, 
                                            test_neg=test_neg, 
                                            train_labels=train_labels,
                                            test_labels=test_labels, 
                                            test_labels_bs=test_labels_bs, 
                                            # pos_answers=p_answer,
                                            # neg_answers=n_answer,
                                            n_probes=args.n_probes,
                                            output_folder=output_folder)
            viz_data[current_run] = viz_method
            #save evals as csv:
            evals_df = pd.DataFrame.from_dict(evals, orient="index", columns=["value"])
            evals_df.to_csv(f"{output_folder}/evals.csv")
            print("evals", evals)
            
        pickle.dump(viz_data, open(f"{output_folder}/viz_data.pickle", "wb"))
        create_accs_visualization_2(viz_data, output_folder=output_folder, n_probes=50)