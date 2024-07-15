from copy import deepcopy
from cluster_norm.utils.CRC import CRC
from cluster_norm.utils.CCS import CCS
import numpy as np

import torch as t
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN

from jaxtyping import Float
from typing import Tuple


def fit_logreg(
        train_pos: Float[Tensor, "batch d_hidden"],
        train_neg: Float[Tensor, "batch d_hidden"],
        train_labels: np.ndarray,

) -> float:
    lr = LogisticRegression(max_iter=10000)
    lr.fit(train_pos-train_neg, train_labels)
    return lr

def fit_ccs(pos, neg, labels, normalize, n_probes=50, device=t.device("cuda")):
    p = deepcopy(pos)
    n = deepcopy(neg)
    
    ccs = CCS(
        pos=p,
        neg=n,
        normalize=normalize,
        n_probe=n_probes,
        device=device
    )
    ccs.optimize()
    
    accs = []
    # for probe in ccs.probes:
    #     preds = ccs.predict(probe, pos, neg)
    #     acc = (preds == labels).mean()
    #     acc = max(acc, 1-acc)
    #     accs.append(acc)
    return accs, deepcopy(ccs)

def fit_crc(pos, neg, normalize):
    p = deepcopy(pos)
    n = deepcopy(neg)
    
    crc = CRC()
    crc.fit(p, n, normalize)
    return crc
    