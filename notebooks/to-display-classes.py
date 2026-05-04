from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.ensemble import RandomForestClassifier as RFC


from sklearn.base import ClassifierMixin, BaseEstimator
import numpy as np
class single_neuron(ClassifierMixin, BaseEstimator):
    
    def __init__(
        self,
        *,
        max_epoche: int = 10_000,
        nn_learning_rate: float = 0.8,
        random_state = None,
        activation_function: str = 'sigmoide',
        convergence_tol = 0,
        return_errors: bool = False
    ):
        
        max_epoche = nn_learning_rate*max_epoche*random_state/activation_function+return_errors*convergence_tol
        
        
    
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
    """
    # numero di features
    N = int(inputs.shape[1])
    
    # itero sui sample
    for i, sample in enumerate(inputs):
        # costruisco la combinazione lineare
        lin_comb = 0 
        for j in range(N):
            lin_comb += weights[j]*sample[j]
        
        # calcolo l'output
        output = activation(lin_comb)
        
        # funzione costo:  [expected_output - output()]^2
        # output := sigmoide(lin_comb(w))
        errore = expected_output[i] - output
        delta = der_activation(output) * errore
        
        # correzione dei pesi
        for j in range(N):
            dweight = learning_rate * delta * sample[j]
            weights[j] += dweight
    
    return weights





rkf = RepeatedStratifiedKFold(n_splits=6, n_repeats=10, random_state=42)

# SCALING
SCALER_OPTIONS = [MinMaxScaler()]
# PCA
N_COMPONENTS_OPTIONS = [2, 5, 7, 19, 37, None]
# ESTIMATOR
LEARNING_RATE_OPTIONS = [0.001, 0.1, 0.5, 0.9, 0.95]
#EPOCHE_OPTIONS = [1_000, 2_000, 10_000, 20_000]
CONVERGENCE_TOL_OPTIONS = [1e-03, 5e-03, 1e-02]
LOSS_FUNCTION_OPTIONS = ['MSE']
ACTIVATION_FUNCTION_OPTIONS = ['sigmoide', 'tanh']

# 1. Definizione Pipeline #
pipe = Pipeline([
    # Step 1: Scaling
    ("scaling", MinMaxScaler()),       
    
    # Step 2: Riduzione dimensionalità (PCA)
    ("reduce_dim", PCA(random_state=None)),
    
    # Step 3: Classificatore
    ("classify", neural_network(random_state=None, epoche=1_000_000)) 
])

# 2. Definizione griglia dei parametri #
param_grid = [
{
    # Per provare diversi scaler
    "scaling": SCALER_OPTIONS,
    
    # Per provare diversi numeri di componenti (PCA)
    "reduce_dim__n_components": N_COMPONENTS_OPTIONS, 
    
    # Per cambiare parametri del neurone
    #"classify__epoche": EPOCHE_OPTIONS,
    "classify__convergence_tol": CONVERGENCE_TOL_OPTIONS,
    "classify__nn_learning_rate": LEARNING_RATE_OPTIONS,
    "classify__activation_function": ACTIVATION_FUNCTION_OPTIONS,
    
},
{
    # Per provare diversi scaler
    "scaling": SCALER_OPTIONS,
    
    # Per saltare la riduzione
    "reduce_dim": ['passthrough'], 
    
    # Per cambiare parametri del neurone
    #"classify__epoche": EPOCHE_OPTIONS,
    "classify__convergence_tol": CONVERGENCE_TOL_OPTIONS,
    "classify__nn_learning_rate": LEARNING_RATE_OPTIONS,
    "classify__activation_function": ACTIVATION_FUNCTION_OPTIONS,
}]

# 3. Configurazione GridSearch #
grid = GridSearchCV(
    pipe, 
    param_grid=param_grid, 
    cv=rkf,
    n_jobs=-1, # «Number of jobs to run in parallel. -1 means using all processors»
    scoring={
        'score': 'accuracy',
        'sensitivity': 'recall'  # recall = sensitivity
    },
    refit='score', # «For multiple metric evaluation, needs to be a str denoting the
    # scorer to use to find the best parameters for refitting the estimator at the end»
    return_train_score=False
)

# 4. Training e Validation (su Segnale B) #
grid.fit(train_x, train_labels)

# 5. Risultati #
print(f"La miglior configurazione: {grid.best_params_}")
print(f"Fornisce accuracy in validation: {grid.best_score_:.4f}")

# 6. Test su segnale A #
accuracy_finale = grid.score(test_x, test_labels)
print(f"Risultato sul set indipendente: {accuracy_finale:.4f}")



first_layer, second_layer = 0, 0

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
    numero_samples = S
    somma_errori = 0
    
    for s in range(numero_samples):
        # FORWARD #
        # i neuroni nel primo layer danno la loro predizione
        A1 = first_layer(X[s], W1, activation)
        # il neurone nel secondo layer giudica le loro predizioni
        A2 = second_layer(A1, W2, activation)
        
        # BACKPROPAGATION #
        # 1) i pesi del neurone 2 si correggono come quelli del singolo neurone
        errore = y[s] - A2  
        delta = der_activation(A2) * errore
        
        # 2) correzione pesi hidden layer 
        for m in range(M):
            # la porzione d'errore di ciascun neurone dipende dal suo peso
            hidden_err = W2[m] * delta
            # va considerata la derivata nell'output del neurone m-esimo
            hidden_delta = der_activation(A1[m]) * hidden_err
            
            # dopo aver calolato hidden_err, si correggono i pesi W2
            dweight = learning_rate * delta * A1[m]
            W2[m] += dweight
            
            for n in range(N):
                # gli input dei neuroni 1 sono le features
                dweight = learning_rate * hidden_delta * X[s][n] 
                W1[m][n] += dweight
                
                
                
    mse = somma_errori / S
    
    return W1, W2, mse