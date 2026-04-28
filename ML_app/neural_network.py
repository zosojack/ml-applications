from __future__ import annotations

import numpy as np
from numba import njit
from cost_functions import sigmoide, sigmoide_deriv, tanh, tanh_deriv

@njit
def SGD_algorithm(
    weights: np.ndarray, 
    inputs: np.ndarray,
    expected_output: np.ndarray,
    learning_rate: float,
    activation: callable,
    der_activation: callable
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
    
    errore = 0
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        # output = sigmoide(lin_comb) NOTE: versione precedente che accettava solo sigmoide
        output = activation(lin_comb)
        
        # funzione costo:  [expected_output - output()]^2
        # output := sigmoide(lin_comb(w))
        errore = expected_output[i] - output
        # delta = output * (1-output) * errore NOTE: versione precedente che accettava solo sigmoide
        delta = der_activation(output) * errore
        
        # correzione dei pesi (srotolato per numba)
        for j in range(N):
            dweight = learning_rate * delta * sample[j]
            weights[j] += dweight
        
    return weights, errore

@njit
def neural_fit(
    epoche: int,
    weights: np.ndarray, 
    inputs: np.ndarray,
    expected_output: np.ndarray,
    nn_learning_rate: float,
    activation: callable,
    der_activation: callable,
) -> np.ndarray:
    
    errors = np.zeros(epoche)
    
    for i in range(epoche):
        weights, last_error = SGD_algorithm(
            weights,
            inputs,
            expected_output,
            learning_rate=nn_learning_rate,
            activation=activation,
            der_activation=der_activation
        )
        
        errors[i] = last_error    
            
    return weights, errors
    

@njit
def neural_predict(
    weights: np.ndarray,
    inputs: np.ndarray,
    activation: callable,
) -> np.ndarray:
    """
    Prediction della neural network
    Da lanciare solo dopo aver trainato il modello e fissato i pesi.
    """
    
    N, M = inputs.shape[1], weights.shape[0]
    
    if N != M:
        raise ValueError(f"Il numero di features ({N})\
            non corrisponde al numero di pesi forniti ({M}).")

    output = np.zeros(inputs.shape[0], dtype=np.int64) # 1 output per ogni sample
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        # output[i] = 0 if sigmoide(lin_comb) < 0.5 else 1 NOTE: VERSIONE PRECEDENTE
        output[i] = 0 if activation(lin_comb) < 0.5 else 1 
        
    return output

@njit
def neural_predict_proba(
    weights: np.ndarray,
    inputs: np.ndarray,
    activation: callable,
) -> np.ndarray:
    """
    Prediction della neural network
    Da lanciare solo dopo aver trainato il modello e fissato i pesi.
    """
    
    N, M = inputs.shape[1], weights.shape[0]
    
    if N != M:
        raise ValueError(f"Il numero di features ({N})\
            non corrisponde al numero di pesi forniti ({M}).")
        
    # 2 output per ogni sample: (1-p, p)
    proba = np.zeros((inputs.shape[0],2), dtype=np.float64) 
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        p = activation(lin_comb)
        proba[i, 0] = 1-p
        proba[i, 1] = p 
        
    return proba

# mean squared error TODO
@njit
def mse_loss_function(out, exp):
    N = out.shape[0]
    mse = 0
    for i in range(N):
        mse += (out[i]-exp[i])**2
    return mse / N

# binary cross-entropy TODO
@njit 
def bce_loss_function(out, exp):
    N = out.shape[0]
    bce = 0
  
    return bce

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class neural_network(ClassifierMixin, BaseEstimator):
    
    # per VotingClassifier
    _estimator_type='classifier'
    
    def __init__(
        self,
        epoche: int = 10000,
        loss: str = None, #'MSE',
        nn_learning_rate: float = 0.8,
        random_state = None,
        activation_function: str = 'sigmoide',
        return_errors: bool = False,
    ):
        # attributi dell'estimator
        self.epoche = epoche
        self.nn_learning_rate = nn_learning_rate        
        self.random_state = random_state
        
        self.activation_function = activation_function
        self.activation = None
        self.der_activation = None
        self.activation_is_assigned_ = False
        
        self.return_errors = return_errors

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
    
    # per VotingClassifier
    def _get_tags(self):
        tags = super()._get_tags()
        tags['binary_only'] = True
        return tags
    
    # per VotingClassifier:
    def _more_tags(self):
        return {'classifier': True}
    
    def _assign_activation(self):
        """ Permette di selezionare tra due diverse funzioni di attivazione """
        if self.activation_function == 'sigmoide':
            self.activation = sigmoide
            self.der_activation = sigmoide_deriv
        elif self.activation_function == 'tanh':
            self.activation = tanh
            self.der_activation = tanh_deriv
        else:
            raise ValueError(f"{self.activation_function} non è una funzione di costo valida\n \
                le opzioni valide sono 'sigmoide' e 'tanh'")
        self.activation_is_assigned_ = True
        
    def fit(self, X, y):
        """ Metodo che si occupa di addestrare il modello """
        
        # assegno la funzione di attivazione dell'estimator
        self._assign_activation()
        
        # controllo validità input e conversione in numpy array (utile per pandas DataFrame)
        X, y = check_X_y(X, y)
        
        # scikit-learn  vuole sapere quante classi sono definite
        self.classes_ = np.unique(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        N = X.shape[1]
        
        # inizializzazione randomica dei pesi
        # NOTE: i parametri appresi vogliono l'underscore finale
        self.weights_ = np.random.rand(N)
        
        # algoritmo di stochastic gradient descent eseguito per ogni epoca
        self.weights_, errors = neural_fit(
            epoche=self.epoche,
            weights=self.weights_,
            inputs=X,
            expected_output=y,
            nn_learning_rate=self.nn_learning_rate,
            activation=self.activation,
            der_activation=self.der_activation
        )
        
        if self.return_errors:
            return errors
        else:
            return self
            
    def predict(self, X):
        """ Metodo che si occupa di classificare """
        # controllo che il modello sia stato fittato
        check_is_fitted(self)
        
        # verifica forma dell'input
        X = check_array(X)
        
        # funzione di predict
        return neural_predict(
            self.weights_, 
            X, 
            activation=self.activation,
        )
    
    # NOTE: FUNZIONA SOLO PER ACTIVATION SIGMOIDE!
    def predict_proba(self, X):
        # controllo che il modello sia stato fittato
        check_is_fitted(self)
        
        # verifica forma dell'input
        X = check_array(X)
        
        # funzione di predict
        return neural_predict_proba(
            self.weights_, 
            X, 
            activation=self.activation,
        )
        