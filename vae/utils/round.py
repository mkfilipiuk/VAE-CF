import numpy as np
from functools import partial


def round_n(x, n=8):
    return n * int(np.ceil(x / n))

round_8 = partial(round_n, n=8)
