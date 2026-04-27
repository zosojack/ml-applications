import numpy as np
from numba import njit

@njit(cache=True)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

@njit(cache=True)
def sigmoide_deriv(output):
    return output * (1.0 - output)

@njit(cache=True)
def tanh(x):
    return np.tanh(x)

@njit(cache=True)
def tanh_deriv(output):
    return 1.0 - output**2