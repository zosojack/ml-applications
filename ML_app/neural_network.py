import numpy as np
from numba import njit
from cost_functions import sigmoide

@njit
def SGD_algorithm(
    weights: np.ndarray, 
    inputs: np.ndarray,
    expected_output: np.ndarray,
    learning_rate: float = 0.9,
) -> np.ndarray:
    """
    Stochastic Gradient Descent Algorithm
    Interessante: l'algoritmo itera sui sample, aggiornando i pesi ad ogni iterazione:
    non attende di conoscere le prestazioni dei pesi iniziali su tutti i sample, bensì
    aggiorna ad ogni step!
    """
    
    N, M = inputs.shape[1], weights.shape[0]
    
    if N != M:
        raise ValueError(f"Il numero di features ({N})\
            non corrisponde al numero di pesi forniti ({M}).")
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        output = sigmoide(lin_comb)
        
        # funzione costo:  [expected_output - output()]^2
        # output = sigmoide(lin_comb(w))
        errore = expected_output[i] - output
        delta = output * (1-output) * errore
        
        # correzione dei pesi
        dweights = learning_rate * delta * sample
        weights += dweights