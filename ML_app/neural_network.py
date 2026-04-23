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
) -> np.ndarray:
    """
    Stochastic Gradient Descent Algorithm
    Interessante: l'algoritmo itera sui sample, aggiornando i pesi ad ogni iterazione:
    non attende di conoscere le prestazioni dei pesi iniziali su tutti i sample, bensì
    aggiorna ad ogni step!
    """
    
    # numero di features
    N, M = int(inputs.shape[1]), int(weights.shape[0])
    
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
        
        # correzione dei pesi (srotolato per numba)
        for j in range(N):
            dweight = learning_rate * delta * sample[j]
            weights[j] += dweight
        
    return weights

@njit
def neural_fit(
    epoche: int,
    weights: np.ndarray, 
    inputs: np.ndarray,
    expected_output: np.ndarray,
    nn_learning_rate: float = 0.9,
) -> np.ndarray:
    
    for i in range(epoche):
            weights = SGD_algorithm(
                weights,
                inputs,
                expected_output,
                learning_rate=nn_learning_rate
            )
            
    return weights
    

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

# mean squared error
@njit
def mse_loss_function(out, exp):
    N = out.shape[0]
    mse = 0
    for i in range(N):
        mse += (out[i]-exp[i])**2
    return mse / N

# binary cross-entropy
@njit 
def bce_loss_function(out, exp):
    N = out.shape[0]
    bce = 0
  
    return bce

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class neural_network(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        epoche: int = 10000,
        loss: str = 'MSE',
        nn_learning_rate: float = 0.8,
        random_state = None
    ):
        # 1. attributi dell'estimator
        self.epoche = epoche
        self.nn_learning_rate = nn_learning_rate        
        self.random_state = random_state
        
        self.loss = loss
        # TODO: implementare la loss function binary cross-entropy
        """
        if loss == 'MSE':
            self.loss = mse_loss_function
        elif loss == 'BCE':
            self.loss = bce_loss_function
        else:
            raise ValueError(f"loss dev'essere 'MSE' o 'BCE', invece è {loss}")
        """
        
        
    def fit(self, X, y):
        # 2. controllo validità input e conversione in numpy array (utile per pandas DataFrame)
        X, y = check_X_y(X, y)
        
        # 3. scikit-learn  vuole sapere quante classi sono definite
        self.classes_ = np.unique(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        N = X.shape[1]
        
        # 4. inizializzazione randomica dei pesi
        # NOTE: i parametri appresi vogliono l'underscore finale
        self.weights_ = np.random.rand(N)
        
        # 5. algoritmo di stochastic gradient descent eseguito per ogni epoca
        self.weights_ = neural_fit(
            epoche=self.epoche,
            weights=self.weights_,
            inputs=X,
            expected_output=y,
            nn_learning_rate=self.nn_learning_rate,
        )
            
        return self
            
    def predict(self, X):
        # 6. controllo che il modello sia stato fittato
        check_is_fitted(self)
        
        # verifica forma dell'input
        X = check_array(X)
        
        # 7. funzione di predict
        return neural_predict(self.weights_, X)