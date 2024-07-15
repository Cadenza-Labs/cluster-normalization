from copy import deepcopy
import numpy as np
import torch as t
from torch import nn, Tensor
import torch.nn.functional as F

from sklearn.decomposition import PCA
from cluster_norm.utils.norm import normalize_burns, normalize_cluster

from jaxtyping import Float


class CRC(object):
    
    def __init__(self, normalize_std=False):
        self.pca = None
        self.normalize_std = normalize_std

    def fit(
        self,
        pos: Float[Tensor, "batch d_hidden"],
        neg: Float[Tensor, "batch d_hidden"],
        normalize: str="burns"
    ) -> float:
        p = deepcopy(pos)
        n = deepcopy(neg)
        
        assert normalize in ["burns", "cluster"]
        f = normalize_burns if normalize == "burns" else normalize_cluster
        
        if f == normalize_burns:
            print("CRC norm burns")
        elif f == normalize_cluster:
            print("CRC norm cluster")
            
        p, n = f(p, n, device=pos.device, std=self.normalize_std)
        pca = PCA(n_components=1)
        self.pca = pca.fit(p - n)

    def predict(self, pos, neg):
        p = deepcopy(pos)
        n = deepcopy(neg)
        
        p, n = normalize_burns(p, n, device=pos.device, std=self.normalize_std)
        preds = self.pca.transform(p - n)
        preds = preds.squeeze(-1) > 0.
        return preds

    def evaluate(self, pos, neg, labels):
        p = deepcopy(pos)
        n = deepcopy(neg)
        
        preds = self.predict(p, n)
        acc = (preds == labels).mean()
        return max(acc, 1-acc)      