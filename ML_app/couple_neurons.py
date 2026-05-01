from __future__ import annotations
import numpy as np
from numba import njit

from ML_app.cost_functions import sigmoide, sigmoide_deriv, tanh, tanh_deriv, ReLU, ReLU_deriv

@njit
def first_layer(
    S1: np.ndarray,
    W1: np.ndarray,
    activation: callable
) -> np.ndarray:
    ''' 
    Un sample per volta, il primo layer processa le features,
    restituisce una predizione per ciascun neaurone.
    '''
    N = S1.shape[0] # numero features
    M = W1.shape[0] # numero neuroni nel primo layer
    
    lin_comb = np.zeros(M) # una comb lin per ciascun neurone
    
    for m in range(M):
        for n in range(N):
            lin_comb[m] += S1[n]*W1[m][n]
    
    A1 = activation(lin_comb) # A1 sarà di simensione M
    return A1

@njit
def second_layer(
    A1: np.ndarray,
    W2: np.ndarray,
    activation: callable
) -> float:
    ''' 
    Il secondo layer assegna importanze diverse ai neuroni del primo,
    restituisce la prediction del neurone finale. 
    '''
    M = A1.shape[0] # tecnicamente, uguale al numero di pesi neurone 2
    
    lin_comb = 0
    
    for m in range(M):
        lin_comb += A1[m]*W2[m]
        
    A2 = activation(lin_comb)
    return A2  

@njit
def SGD(
    X: np.ndarray, # (n_samples, n_features)
    y: np.ndarray,
    W1: np.ndarray, # (hidden_neurons, n_features)
    W2: np.ndarray, # (hidden_neurons, 1)
    learning_rate: float,
    activation: callable,
    der_activation: callable,
) -> tuple[np.ndarray, np.ndarray, float]:
    '''
    * Stochastic Gradient Descent Algorithm *
    Un sample alla volta, primo e secondo layer valutano le features.
    Prima di passare al prossimo sample, i pesi vengono ritarati.
    Tutorial bakpropagation per due layer:
    https://medium.com/@hoangngbot/code-a-2-layer-neural-network-from-scratch-33d7db0f0e5f
    '''
    S, N = X.shape # samples, features
    M = W2.shape[0] # numero neuroni layer 1
    
    somma_errori = 0
    
    for s in range(S):
        # FORWARD #
        # i neuroni nel primo layer danno la loro predizione
        A1 = first_layer(X[s], W1, activation)
        # il neurone nel secondo layer giudica le loro predizioni
        A2 = second_layer(A1, W2, activation)
        
        # BACKPROPAGATION #
        # 1) i pesi del neurone 2 si correggono come quelli del singolo neurone
        errore = y[s] - A2  
        delta = der_activation(A2) * errore
        somma_errori += errore**2
        
        # 2) correzione pesi hidden layer 
        for m in range(M):
            # la porzione d'errore di ciascun neurone dipende dal suo peso
            hidden_err = W2[m] * delta
            # va considerata la derivata nell'output del neurone m-esimo
            hidden_delta = der_activation(A1[m]) * hidden_err
            
            # dopo aver calolato hidden_err, posso correggere i pesi W2
            dweight = learning_rate * delta * A1[m] # input del neurone 1 sono le predizioni A1!
            W2[m] += dweight
            
            for n in range(N):
                # gli input dei neuroni 1 sono le features!
                dweight = learning_rate * hidden_delta * X[s][n] 
                W1[m][n] += dweight
                
    mse = somma_errori / S
    
    return W1, W2, mse

@njit
def check_convergence_patience(
    mse_now: float,
    mse_before: float,
    tol: float,
    patience: int
) -> int:
    """
    Tutorial convergence patience-based:
    https://medium.com/@mbonsign/iterative-refinement-breaking-through-convergence-plateaus-in-neural-language-models-f8eb03e04cb7
    """
    diff = (mse_before - mse_now) / mse_before
    
    if diff < tol: # se migliora meno del valore di tolleranza
        patience += 1
    else:
        patience = 0 # Reset se il modello sta ancora imparando
        
    return patience

@njit
def neural_fit(
    max_epoche: int,
    convergence_tol: float, 
    X: np.ndarray, # (n_samples, n_features)
    y: np.ndarray,
    W1: np.ndarray, # (hidden_neurons, n_features)
    W2: np.ndarray, # (hidden_neurons, 1)
    learning_rate: float,
    activation: callable,
    der_activation: callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Esegue l'algoritmo SGD fino a raggiungimento della convergenza, oppure fino ad un endstep.
    """
    
    # Inizializzazione variabili utili alla logica di convergenza
    mse = np.zeros((max_epoche // 1000) + 1) # un valore ogni mille
    convergence_idx = max_epoche
    check_step = 500 # controllo convergenza ogni 500 epoche
    mse_before = 1e10
    patience = 0
    
    
    # Ciclo reale
    for i in range(max_epoche):
        W1, W2, mse_now = SGD(
            X=X,
            y=y,
            W1=W1,
            W2=W2,
            learning_rate=learning_rate,
            activation=activation,
            der_activation=der_activation
        )
        
        if i % check_step == 0:
            patience = check_convergence_patience(
                mse_now,
                mse_before,
                convergence_tol*learning_rate, # minore il LR, più pazienza è necessaria
                patience
            )
            mse_before = mse_now
            # Se il modello non impara per tre controlli consecutivi, interrompe il fit
            if patience >= 3:
                convergence_idx = i
                break

        # Conserva un errore ogni mille         
        if i % 1000 == 0:
            mse[i // 1000] = mse_now  
        
    return W1, W2, mse, convergence_idx


@njit
def neural_predict(
    W1: np.ndarray,
    W2: np.ndarray,
    X: np.ndarray,
    activation: callable
) -> np.ndarray:
    """
    Prediction della neural network.
    Da lanciare solo dopo aver trainato il modello e fissato i pesi.
    """
    S = X.shape[0]
    prediction = np.zeros(S)
    
    for s in range(S):
        A1 = first_layer(X[s], W1, activation)
        A2 = second_layer(A1, W2, activation)
        prediction[s] = 0 if A2 < 0.5 else 1
        
    return prediction

@njit
def neural_predict_proba(
    W1: np.ndarray,
    W2: np.ndarray,
    X: np.ndarray,
    activation: callable
) -> np.ndarray:
    """
    Prediction della neural network.
    Da lanciare solo dopo aver trainato il modello e fissato i pesi.
    Restituisce la probabilità della predizione. 
    """
    S = X.shape[0]
    prediction = np.zeros((S,2))
    
    for s in range(S):
        A1 = first_layer(X[s], W1, activation)
        A2 = second_layer(A1, W2, activation)
        prediction[s][0], prediction[s][1]  = 1-A2, A2
        
    return prediction

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# TODO: se si vuole, implementare diverse loss function

class couple_neurons(ClassifierMixin, BaseEstimator):
    
    # per VotingClassifier
    _estimator_type='classifier'
    
    def __init__(
        self,
        *,
        hidden_neurons: int = 2,
        epoche: int = 10_000,
        nn_learning_rate: float = 0.8,
        random_state = None,
        activation_function: str = 'sigmoide',
        return_errors: bool = False,
        convergence_tol = 0,
    ):
        # attributi dell'estimator
        self.hidden_neurons = hidden_neurons # neuroni nell'hidden layer
        self.epoche = epoche
        self.nn_learning_rate = nn_learning_rate        
        self.random_state = random_state
        
        self.activation_function = activation_function
        self.activation = None
        self.der_activation = None
        self.activation_is_assigned_ = False
        
        self.return_errors = return_errors
        self.convergence_tol = convergence_tol
        
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
        elif self.activation_function == 'ReLU':
            self.activation = ReLU
            self.der_activation = ReLU_deriv
        else:
            raise ValueError(f"{self.activation_function} non è una funzione di costo valida\n \
                le opzioni valide sono 'sigmoide', 'tanh' e 'ReLU'")
        self.activation_is_assigned_ = True
        
    def fit(self, X, y):
        """ Metodo che si occupa di addestrare il modello """
        # assegno la funzione di attivazione dell'estimator
        self._assign_activation()
        
        # controllo validità input e conversione in numpy array (utile per pandas DataFrame)
        X, y = check_X_y(X, y)
        
        # scikit-learn  vuole sapere quante classi sono definite
        self.classes_ = np.unique(y)
        
        # assegno seed se fornito in costruzione
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # il numero di pesi dei neuroni nel layer 1 è il numero di features
        N = X.shape[1]
        self.W1_ = np.random.rand(self.hidden_neurons, N)
        # il numero di pesi del neurone nel layer 2 è il numero di neuroni nell'1
        self.W2_ = np.random.rand(self.hidden_neurons)
        
        self.W1_, self.W2_, errors, convergence_idx = neural_fit(
            max_epoche=self.epoche,
            convergence_tol=self.convergence_tol,
            W1=self.W1_,
            W2=self.W2_,
            X=X,
            y=y,
            learning_rate=self.nn_learning_rate,
            activation=self.activation,
            der_activation=self.der_activation,
        )
        
        # logica di convergenza
        self._has_converged = True if convergence_idx < self.epoche else False
        self._convergence_idx = convergence_idx # inizializzato a epoche
            
        # immaganizzazione degli errori
        if self.return_errors:
            self.errors_ = errors[:int(self._convergence_idx/1000)]
            
        return self
    
    def predict(self, X):
        """ Metodo che si occupa di classificare """
        # controllo che il modello sia stato fittato
        check_is_fitted(self)
        
        # verifica forma dell'input
        X = check_array(X)
        
        # funzione di predict
        return neural_predict(
            self.W1_, 
            self.W2_,
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
            self.W1_,
            self.W2_, 
            X, 
            activation=self.activation,
        )
    
    