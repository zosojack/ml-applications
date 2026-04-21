import numpy as np
from numba import njit

@njit
def sigmoide(x):
    return 1 / (1 + np.exp(-x))