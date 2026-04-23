from __future__ import annotations

import numpy as np
from numba import njit
from cost_functions import sigmoide

# per parametrizzarlo:
# quante epoche?
# quanto learning rate?
# hai usato ending step o criterio di convergenza?

@njit
def SGD_algorithm(
    weights: np.ndarray, 
    inputs: np.ndarray,
    expected_output: np.ndarray,
    learning_rate: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Gradient Descent Algorithm
    Interessante: l'algoritmo itera sui sample, aggiornando i pesi ad ogni iterazione:
    non attende di conoscere le prestazioni dei pesi iniziali su tutti i sample, bensì
    aggiorna ad ogni step!
    """
    
    # numero di features
    N, M = int(inputs.shape[1]), int(weights.shape[0])
    # numeor di samples
    errors = np.zeros(int(inputs.shape[0]))
    
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
        
        errors[i] = errore
        
    return (weights, errors)

@njit
def neural_predict(
    weights: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """
    Prediction della neural network
    Da lanciare solo dopo aver trainato il modello e fissato i pesi.
    """
    
    N, M = inputs.shape[1], weights.shape[0]
    
    if N != M:
        raise ValueError(f"Il numero di features ({N})\
            non corrisponde al numero di pesi forniti ({M}).")

    output = np.zeros(inputs.shape[0]) # 1 output per ogni sample
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        #output[i] = sigmoide(lin_comb)
        output[i] = 0 if sigmoide(lin_comb) < 0.5 else 1
        
    return output



from sklearn.base import ClassifierMixin, BaseEstimator

class neural_network(BaseEstimator):
    
    def __init__(
        self,
        *,
        epoche: int = 10_000,
        loss: str = 'MSE',
        nn_learning_rate: float = 0.8,
        random_state = None,
        weights = None
    ):
        
        self.epoche = epoche,
        self.nn_learning_rate = nn_learning_rate,        
        
        # TODO: AGGIUNGERE LOSS FUNCTION LOGARITMICA
        """
        a = 0
        if loss == 'MSE':
            # definisci loss_func mse
            a = 1
            self.loss = loss,
        elif loss == 'log':
            # definisci loss_func mse
            a = 2
        else:
            # dai errore
            raise ValueError(f"loss può essere 'MSE' o 'log', invece è {loss}")
        """
        
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            
        self.loss = loss
        self.weights = weights
    
    def fit(self, X, y=None):
        # inizializza pesi
        N = X.shape[1]
        self.weights = np.random.rand(N)
        
        for i in range(self.epoche):
            self.weights = SGD_algorithm(
                self.weights,
                X,
                y,
                learning_rate = self.nn_learning_rate
            )
            
        self.is_fitted_ = True
        return self
            
    def predict(self, X):
        return neural_predict
            
        
"""       
...     def __init__(self, *, param=1):
...         self.param = param
...     def fit(self, X, y=None):
...         self.is_fitted_ = True
...         return self
...     def predict(self, X):
...         return np.full(shape=X.shape[0], fill_value=self.param)
"""