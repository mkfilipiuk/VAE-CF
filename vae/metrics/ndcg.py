"""
Discounted Cumulative Gain @ R is

    DCG@R(u,ω) := Σ_{r=1}^{R} I[ω(r) ∈ I_u] − 1 / log(r + 1) / IDCG@R(u,ω)
    IDCG@R(u,ω) := Σ_{r=1}^{|I_u|} 1 / log(r + 1)

https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
https://arxiv.org/pdf/1802.05814.pdf, chapter 4.2
"""

import numpy as np
from scipy.sparse import csr_matrix


def ndcg(X_true: csr_matrix, X_top_k: np.array, R=100) -> np.array:
    """ Calculate ndcg@R for each users in X_true and X_pred matrices

    Args:
        X_true: Matrix containing True values for user-item interactions
        X_top_k: Matrix containing inidices picked by model
        R: Number of elements taken into consideration

    Returns:
        Numpy array containing calculated ndcg@R for each user
    """

    penalties = 1. / np.log2(np.arange(2, R + 2))
    selected = np.take_along_axis(X_true, X_top_k[:, :R], axis=-1)

    DCG = selected * penalties

    cpenalties = np.empty(R + 1)
    np.cumsum(penalties, out=cpenalties[1:])
    cpenalties[0] = 0
    maxhit = np.minimum(X_true.getnnz(axis=1), R)
    IDCG = cpenalties[maxhit]

    return DCG / IDCG
