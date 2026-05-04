import numpy as np
from numba import njit

@njit(cache=True)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

@njit(cache=True)
def sigmoide_deriv(s):
    return s * (1.0 - s)

@njit(cache=True)
def tanh(x):
    return np.tanh(x)

@njit(cache=True)
def tanh_deriv(x):
    return 1.0 - x**2

@njit(cache=True)
def ReLU(x):
    return np.maximum(0.0, x)

@njit(cache=True)
def ReLU_deriv(x):
    return 0.0 if x <= 0 else 1.0