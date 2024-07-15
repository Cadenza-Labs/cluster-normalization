import numpy as np
import torch as t
from torch import nn, Tensor
import torch.nn.functional as F

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from cluster_norm.utils.norm import normalize_burns, normalize_cluster

from functools import partial
from jaxtyping import Float
from typing import Tuple, Optional
from copy import deepcopy
from tqdm import trange


class CCS(object):

    def __init__(self,
                 pos: Float[t.Tensor, "batch d_hidden"],
                 neg: Float[t.Tensor, "batch d_hidden"],
                 normalize: str="burns",
                 lr: float=0.01,
                 weight_decay: float=0.,
                 batch_size: int=128,
                 n_epoch: int=1000,
                 n_probe: int=50,
                 device: str="cuda"):

        self.device = t.device(device)
        assert normalize in ["burns", "cluster"]
        self.normalize = normalize_burns if normalize == "burns" else normalize_cluster
        print("CCS norm", "burns" if self.normalize == normalize_burns else "cluster")
        
        p, n = deepcopy(pos), deepcopy(neg)
        self.train_pos, self.train_neg = self.normalize(p, n, self.device)
        self.d = p.shape[-1]

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.n_probe = n_probe

        self.probes = []
        self.probe = None
        self.best_probe = None

    def init_probe(self):

        linear = nn.Linear(self.d, 1)
        t.nn.init.trunc_normal_(linear.weight, std=1./np.sqrt(self.d))
        t.nn.init.zeros_(linear.bias)
        self.probe = nn.Sequential(
            linear,
            nn.Sigmoid()
        ).to(self.device)

    def get_loss(self,
                 p_pos: Float[t.Tensor, "batch"],
                 p_neg: Float[t.Tensor, "batch"]
    ) -> Float[t.Tensor, "1"]:
        
        l_consistency = ((p_pos - (1 - p_neg))**2).mean()
        # symmetric confidence loss
        l_confidence = [t.min(p_pos, p_neg), t.min(1-p_pos, 1-p_neg)]
        l_confidence = (t.min(*l_confidence)**2).mean()
        return l_consistency + l_confidence

    def train(self) -> float:
        '''
        Fit a single probe.
        For CCS, we fit multiple probes during `optimize`, choosing the best.
        '''
        pos, neg = self.train_pos, self.train_neg

        # training setup
        opt = t.optim.AdamW(self.probe.parameters(),
                            lr = self.lr,
                            weight_decay=self.weight_decay)
        batch_size = len(pos) if self.batch_size == -1 else self.batch_size
        n_batch = len(pos) // batch_size

        # train probe
        for epoch in range(self.n_epoch):
            for i in range(n_batch):
                pos_b = pos[i*batch_size:(i+1)*batch_size]
                neg_b = neg[i*batch_size:(i+1)*batch_size]
                p_pos, p_neg = self.probe(pos_b), self.probe(neg_b)
                loss = self.get_loss(p_pos, p_neg)
                opt.zero_grad()
                loss.backward()
                opt.step()
        # save trained probe
        self.probes.append(deepcopy(self.probe))
        # return final loss
        return loss.detach().cpu().item()

    def optimize(self):

        best_loss = float("inf")
        for _ in trange(self.n_probe, desc="fitting probes"):
            self.init_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = deepcopy(self.probe)
                best_loss = loss

    def predict(self,
                probe,
                pos: Float[t.Tensor, "batch d_hidden"],
                neg: Float[t.Tensor, "batch d_hidden"],
                normalize: bool=True
    ) -> np.ndarray:
        '''
        Predict on a batch, for a given probe.
        '''
        p = deepcopy(pos)
        n = deepcopy(neg)
        
        # Default norm is burns on inference, as in original CCS paper
        if normalize: p, n = normalize_burns(p, n, self.device)
        else: p, n = p.to(self.device), n.to(self.device)
        with t.no_grad():
            p_pos, p_neg = probe(p), probe(n)
        p = 0.5*(p_pos + (1 - p_neg))
        pred = (p.detach().cpu().numpy() < 0.5)[:, 0].astype(int)
        return pred