import numpy as np
from time import time
from scipy.sparse import csr_matrix
from functools import partial
import multiprocessing

GLOBAL_TOP_K_SORTED_BUFFER = None
GLOBAL_TOP_K_SORTED_STEP = 20
GLOBAL_TOP_K_SORTED_R = 100
def get_top_k_sorted_worker(i):
    sub_arr = -GLOBAL_TOP_K_SORTED_BUFFER[i:i+GLOBAL_TOP_K_SORTED_STEP]
    arg_prt = np.argpartition(sub_arr, GLOBAL_TOP_K_SORTED_R)[:, :GLOBAL_TOP_K_SORTED_R]
    val_prt = np.take_along_axis(sub_arr, arg_prt, axis=-1)
    srt_prt = np.argsort(val_prt)
    return np.take_along_axis(arg_prt, srt_prt, axis=-1) 


def get_top_k_sorted(X: np.array, R=100, n_workers=16) -> np.array:
    global GLOBAL_TOP_K_SORTED_BUFFER
    global GLOBAL_TOP_K_SORTED_R
    GLOBAL_TOP_K_SORTED_BUFFER = X
    GLOBAL_TOP_K_SORTED_R = R
    with multiprocessing.Pool(n_workers) as p:
        return np.concatenate(p.map(get_top_k_sorted_worker, range(0, X.shape[0], GLOBAL_TOP_K_SORTED_STEP)))
