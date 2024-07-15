import pandas as pd
import numpy as np

path = "/home/wombat_share/laurito/cluster_norm/runs/experiment_2/mistral/imdb/layer23/ccs_cluster.csv"
df = pd.read_csv(path)

# dividie by biased and unbiased and store in two different df
# the first 100 elements of the csv are unbiased
df_unbiased = df.iloc[:100]
df_biased = df.iloc[100:]

def calc_stats(group):
    mean = np.mean(group['accuracy'])
    std = np.std(group['accuracy'], ddof=1)  # ddof=1 for sample standard deviation
    return pd.Series({'mean_accuracy': mean, 'std_accuracy': std})

# Apply the function to each template group
grouped_unbiased = df_unbiased.groupby('template').apply(calc_stats).reset_index()
grouped_biased = df_biased.groupby('template').apply(calc_stats).reset_index()

print("unbiased", grouped_unbiased)
print("biased", grouped_biased)
