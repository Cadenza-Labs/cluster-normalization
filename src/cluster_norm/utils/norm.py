import numpy as np

from torch import Tensor

from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN

from jaxtyping import Float
from typing import Tuple


def normalize_burns(
        x: Float[Tensor, "batch d_hidden"],
        y: Float[Tensor, "batch d_hidden"],
        device: str,
        std: bool=False
) -> Tuple[Float[Tensor, "batch d_hidden"], Float[Tensor, "batch d_hidden"]]:

    m_x, m_y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
    x = x - m_x
    y = y - m_y
    
    if std:
        s_x, s_y = x.std(dim=0, keepdim=True), y.std(dim=0, keepdim=True)
        x = x / s_x
        y = y / s_y
    
    return x.to(device), y.to(device)


def normalize_cluster(
        x: Float[Tensor, "batch d_hidden"],
        y: Float[Tensor, "batch d_hidden"],
        device: str,
        std: bool=False,
        verbose: bool=False,
) -> Tuple[Float[Tensor, "batch d_hidden"], Float[Tensor, "batch d_hidden"]]:
    
    # NOTE: is mean the best option here? what about concat or diff? 
    # prob doesn't make a difference but haven't tried
    v = (x + y) / 2.

    hdb = HDBSCAN(min_cluster_size=5, metric="euclidean")
    hdb.fit(v)
    if verbose: print(f"\n{len([x for x in set(hdb.labels_) if x >= 0])} clusters found.\n")

    global_m_x, global_m_y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
    global_s_x, global_s_y = x.std(dim=0, keepdim=True), y.std(dim=0, keepdim=True)
    for label in set(hdb.labels_):
        ixs = np.where(hdb.labels_ == label)
        # for outliers, normalize globally
        if label < 0:
            m_x, m_y = global_m_x, global_m_y
            s_x, s_y = global_s_x, global_s_y
        else:
            m_x, m_y = x[ixs].mean(dim=0, keepdim=True), y[ixs].mean(dim=0, keepdim=True)
            s_x, s_y = x[ixs].std(dim=0, keepdim=True), y[ixs].std(dim=0, keepdim=True)
        x[ixs] = (x[ixs] - m_x)
        y[ixs] = (y[ixs] - m_y)

        if std:
            x[ixs] = x[ixs] / s_x
            y[ixs] = y[ixs] / s_y

    return x.to(device), y.to(device)