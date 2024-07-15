
import os
import pickle
from pprint import pp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def extract_last_word(text):
    # Split by period and filter out any empty segments
    segments = [segment.strip() for segment in text.split('.') if segment.strip()]
    # Take the last segment and split it by spaces to get words
    last_words = segments[-1].split() if segments else []
    # Return the last word or None if there are no words
    return last_words[-1] if last_words else None

def zero_shot(pos_answers, neg_answers, labels):
    zero_shot_accs = [] 
    for logits in [pos_answers, neg_answers]:
        # TODO: That's only imdb specific right now
        llm_predictions = (logits == "positive")
        zero_shot_accs.append((llm_predictions == labels).mean())
    
    print("zero_shot_accs", zero_shot_accs)
    return np.max(zero_shot_accs, axis=0)


def _get_def(true_accs, biased_accs):
    df = pd.DataFrame(columns=["accuracy", "template"])
    accuracies, templates = [], []
    for idx, _ in enumerate(true_accs):
        true_acc = true_accs[idx]
        biased_acc = biased_accs[idx]

        accuracies.extend([true_acc, biased_acc])
        templates.extend([f"Default", f"Random Word"])
    
    df["accuracy"] = accuracies
    df["template"] = templates
    
    return df
    

def create_accs_visualization_2(viz_data, output_folder, n_probes=50):
    pp(viz_data)
    non_bs_sent_accs_burns, non_bs_accs_burns = viz_data["non_bs"]["burns"] 
    bs_sent_accs_burns, bs_accs_burns = viz_data["bs"]["burns"]
    
    non_bs_sent_accs_cluster, non_bs_accs_cluster = viz_data["non_bs"]["cluster"]
    bs_sent_accs_cluster, bs_accs_cluster = viz_data["bs"]["cluster"]
    
    sent_accs_burns = np.concatenate([non_bs_sent_accs_burns, bs_sent_accs_burns])
    bs_accs_burns = np.concatenate([non_bs_accs_burns, bs_accs_burns])
    
    sent_accs_cluster = non_bs_sent_accs_cluster + bs_sent_accs_cluster
    bs_accs_cluster = non_bs_accs_cluster + bs_accs_cluster
    
    burns = _get_def(sent_accs_burns, bs_accs_burns)
    cluster = _get_def(sent_accs_cluster, bs_accs_cluster)    
    
    colours = {"Default:GT": "blue", "Default:Random": "lightblue",
                    "Random:GT": "red", "Random:Random": "lightcoral"}
    legend_elements = [
        mpatches.Patch(facecolor="red", label="Random:GT"),
        mpatches.Patch(facecolor="lightcoral", label="Random:Random"),
        mpatches.Patch(facecolor="blue", label="Default:GT"),
        mpatches.Patch(facecolor="lightblue", label="Default:Random")
    ]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, squeeze=True, figsize=(10, 3))
    for ax, df, label in zip(axs, [burns, cluster], ["Burns-Norm", "Cluster-Norm"]):
        df["template"] = ["Default:GT", "Default:Random"]*n_probes + ["Random:GT", "Random:Random"]*n_probes
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
    plt.savefig(f"{output_folder}/ccs.png")
    print("saved plot to")
    


def create_accs_visualization(viz_data, method, output_folder, n_probes=50):
    sent_accs_burns, bs_accs_burns = viz_data["burns"]
    sent_accs_cluster, bs_accs_cluster = viz_data["cluster"]

    df_burns = pd.DataFrame(columns=["setting", "accuracy"])
    df_burns["setting"] = ["correct label"]*n_probes + ["random word"]*n_probes
    df_burns["accuracy"] = sent_accs_burns + bs_accs_burns
    
    df_cluster = pd.DataFrame(columns=["setting", "accuracy"])
    df_cluster["setting"] = ["correct label"]*n_probes + ["random word"]*n_probes
    df_cluster["accuracy"] = sent_accs_cluster + bs_accs_cluster
    
    image_file = f"{output_folder}/{method}_accs.png"
    colours = sns.color_palette("Set1", n_colors=2)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Correct Label',
                          markerfacecolor=colours[0], markersize=10),
                 plt.Line2D([0], [0], marker='o', color='w', label='Random Word',
                          markerfacecolor=colours[1], markersize=10)]    
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, squeeze=True, figsize=(10, 3))
    for ax, df, label in zip(axs, [df_burns, df_cluster], ["Burns-Norm", "Cluster-Norm"]):
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
    if os.path.exists(image_file):
        # just in case, delete the existing file before saving
        os.remove(image_file)
        
    plt.savefig(image_file)    
    

def evaluate_ccs(ccs, test_pos, test_neg, test_labels_sentiment, test_labels_bs):
    sent_accs, bs_accs = [], []
    for probe in ccs.probes:
        preds = ccs.predict(probe, test_pos, test_neg, True)
        sent_acc = (preds == test_labels_sentiment).mean()
        sent_acc = max(sent_acc, 1-sent_acc)
        bs_acc = (preds == test_labels_bs).mean()
        bs_acc = max(bs_acc, 1-bs_acc)
        sent_accs.append(sent_acc)
        bs_accs.append(bs_acc)
        
    return sent_accs, bs_accs
    
