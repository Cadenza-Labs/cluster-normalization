import os, sys, gc, pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import torch as t

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cluster_norm.utils.CCS import CCS
from cluster_norm.utils.CRC import CRC

from copy import deepcopy

runs_folder = "/home/wombat_share/laurito/cluster_norm/runs/experiment_2/"
experiment2_path = "/home/wombat_share/laurito/cluster_norm/src/cluster_norm/_experiment2"
activations_path = f"{experiment2_path}/activations"

prompts = pd.read_json(f"{experiment2_path}/prompts_positive.jsonl", 
                            orient="records", lines=True)
true_labels = prompts["label"].values == "positive"
alice_labels = prompts["alice"].values == "positive"
N_PROBES = 50

variants = [1, 2]

def pca_plot():
    # Create PCA plot
    pca_legend_elements = [
    mpatches.Patch(facecolor="moccasin", label="TP:AN"),
    mpatches.Patch(facecolor="lightblue", label="TN:AN"),
    mpatches.Patch(facecolor="orange", label="TP:AP"),
    mpatches.Patch(facecolor="blue", label="TN:AP")
    ]
    pca_colour_map = {
        (1, 0): "moccasin", # true positive, alice negative
        (0, 0): "lightblue", # true negative, alice negative
        (1, 1): "orange", # true positive, alice positive
        (0, 1): "blue", # true negative, alice positive
    }
    colours = [pca_colour_map[(l1, l2)] for l1, l2 in zip(true_labels, alice_labels)]
    # PCA
    print("PCA")

    fig = plt.figure(figsize=(7.5, 4))
    titles = ["Default", "Alice-opinion"]
    for variant in [1, 2]:
        pos = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_positive.pt")
        neg = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_negative.pt")
        pca = PCA(n_components=3)
        X = pca.fit_transform(pos - neg)
        ax = fig.add_subplot(1, 2, variant, projection="3d")
        ax.view_init(elev=60, azim=45)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, c=colours)
        ax.set_xlabel("PC 1", weight="bold"); ax.set_ylabel("PC 2", weight="bold")
        ax.set_title(titles[variant-1], y=0, pad=-25, verticalalignment="top")
        plt.suptitle("PCA Visualisations")
        plt.figlegend(handles=pca_legend_elements)
        plt.tight_layout()
        plt.savefig(f"{outpath}/pca.png")
    

def get_logistic_regression(perm, layer):
    # Logistic Regression
    accuracies, templates = [], []
    for variant in variants:
        print(f"\n{variant}")
        pos = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_positive.pt")[perm]
        neg = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_negative.pt")[perm]
        split = int(0.7*len(pos))
        X_train, X_test = (pos-neg)[:split], (pos-neg)[split:]
        y_train_true, y_test_true = true_labels[perm][:split], true_labels[perm][split:]
        _, y_test_alice = alice_labels[perm][:split], alice_labels[perm][split:]
        
        lr = LogisticRegression(max_iter=10000)
        lr.fit(X_train, y_train_true)
        acc_true = lr.score(X_test, y_test_true)
        acc_alice = lr.score(X_test, y_test_alice)
        
        print("acc_true", acc_true)
        print("acc_alice", acc_alice)
        t.save(lr, f"{outpath}/lr.pt")

        accuracies.extend([acc_true, acc_alice])
        templates.extend([f"Default", f"Alice-opinion"])
        t.save(lr, f"{outpath}/lr_{variant}.pt")
    df = pd.DataFrame()
    df["accuracy"] = accuracies
    df["template"] = templates
    print("lr", df)
    return df
        
def get_crc(normalize, perm, layer):
    accuracies, templates = [], []
    for variant in variants:
        pos = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_positive.pt")[perm]
        neg = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_negative.pt")[perm]
        split = int(0.7*len(pos))
        
        crc = CRC()
        crc.fit(pos[:split], neg[:split], normalize)
        preds = crc.predict(pos[split:], neg[split:])
        true_acc = (preds == true_labels[perm][split:]).mean()
        if (1-true_acc) > true_acc: true_acc = 1-true_acc
        print("crc true_acc", true_acc, "variant", variant)

        alice_acc = (preds == alice_labels[perm][split:]).mean()
        if (1-alice_acc) > alice_acc: alice_acc = 1-alice_acc
        print("crc alice_acc", alice_acc, "variant", variant)
        
        accuracies.extend([true_acc, alice_acc])
        templates.extend([f"Default", f"Alice-opinion"])
        # save crc
        t.save(crc, f"{outpath}/crc_{normalize}_{variant}.pt")
            
    df = pd.DataFrame()
    df["template"] = templates
    df["accuracy"] = accuracies
    print("crc", df)
    return df

def get_ccs(normalize, perm, layer):
    accuracies, templates = [], []
    for variant in variants:
        pos = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_positive.pt")[perm]
        neg = t.load(f"{activations_path}/{model}/{layer}/variant{variant}_negative.pt")[perm]
        split = int(0.7*len(pos))
        
        # check if ccs already exists
        if os.path.exists(f"{outpath}/ccs_{normalize}_{variant}.pt"):
            ccs = t.load(f"{outpath}/ccs_{normalize}_{variant}.pt")
        else:
            ccs = CCS(
                pos=deepcopy(pos[:split]),
                neg=deepcopy(neg[:split]),
                normalize=normalize,
                n_probe=N_PROBES,
                device=t.device("cuda:4")
            )
            ccs.optimize()
            t.save(ccs, f"{outpath}/ccs_{normalize}_{variant}.pt")
        
        for probe in ccs.probes:
            preds = ccs.predict(probe, deepcopy(pos[split:]), deepcopy(neg[split:]))
            true_acc = (preds == true_labels[perm][split:]).mean()
            if (1-true_acc) > true_acc: true_acc = 1-true_acc
            alice_acc = (preds == alice_labels[perm][split:]).mean()
            if (1-alice_acc) > alice_acc: alice_acc = 1-alice_acc
            accuracies.extend([true_acc, alice_acc])
            templates.extend([f"Default", f"Alice-opinion"])
        
         
    df = pd.DataFrame()
    df["template"] = templates
    df["accuracy"] = accuracies
    print("ccs" , df)
    return df

# write python main:
if __name__ == "__main__":    
    perm = np.random.permutation(len(true_labels))
    
    for model in ["llama"]: # , 
        layers = os.listdir(f"{activations_path}/{model}/")
        print("layers", layers)
        for layer in layers:
            print("Layer:", layer)
            outpath = f"{runs_folder}/{model}/imdb/{layer}"
            Path(outpath).mkdir(exist_ok=True, parents=True)
            
            #### Cacl results for methods
            lr_results = get_logistic_regression(perm, layer)
            lr_results.to_csv(f"{outpath}/lr.csv", index=False)
            
            crc_results_burns = get_crc("burns", perm, layer)
            crc_results_burns.to_csv(f"{outpath}/crc_burns.csv", index=False)

            crc_results_cluster = get_crc("cluster", perm, layer)
            crc_results_cluster.to_csv(f"{outpath}/crc_cluster.csv", index=False)
            
            ccs_results_cluster = get_ccs("cluster", perm, layer)
            ccs_results_cluster.to_csv(f"{outpath}/ccs_cluster.csv", index=False)
            
            ccs_results_burns = get_ccs("burns", perm, layer)
            ccs_results_burns.to_csv(f"{outpath}/ccs_burns.csv", index=False)

            
            #### Draw pca plot
            pca_plot()

            #### Draw violin plots for CCS
            colours = {"Default:GT": "blue", "Default:Alice": "lightblue",
                    "Alice:GT": "red", "Alice:Alice": "lightcoral"}
            legend_elements = [
                mpatches.Patch(facecolor="red", label="Alice:GT"),
                mpatches.Patch(facecolor="lightcoral", label="Alice:Alice"),
                mpatches.Patch(facecolor="blue", label="Default:GT"),
                mpatches.Patch(facecolor="lightblue", label="Default:Alice")
            ]

            fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, squeeze=True, figsize=(10, 3))
            for ax, df, label in zip(axs, [ccs_results_burns, ccs_results_cluster], ["Burns-Norm", "Cluster-Norm"]):
                df["template"] = ["Default:GT", "Default:Alice"]*N_PROBES + ["Alice:GT", "Alice:Alice"]*N_PROBES
                sns.stripplot(x="template", y="accuracy", data=df, jitter=0.05, ax=ax, palette=colours, hue="template")
                sns.violinplot(x="template", y="accuracy", data=df, ax=ax, palette=colours, hue="template",
                            cut=0, inner=None, width=0.3, alpha=0.5)
                ax.set_xticks([], [])
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel(label)
            axs[0].set_ylabel("Accuracy")
            plt.figlegend(handles=legend_elements, loc="lower left", ncol=2, labelspacing=0.1,
                        borderaxespad=5., frameon=False, fontsize='medium')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"{outpath}/ccs.png")
            
