import numpy as np
from numba import njit 
from ML_app.single_neuron import single_neuron
from ML_app.couple_neurons import couple_neurons
import time


def evaluate_lr_tol_convergence_couple(
    hidden_neurons: int,
    max_epoche: int,
    lr: float,
    tol: float,
    train_x: np.ndarray,
    train_labels: np.ndarray
):
    ''' Per analizzare la convergenza al variare del learning rate e la tolleranza '''
    
    nn = couple_neurons(
        hidden_neurons=hidden_neurons,
        epoche = max_epoche,
        nn_learning_rate = lr,
        activation_function = 'sigmoide',
        convergence_tol=tol
    )    
    
    start_time = time.time()
    
    nn.fit(train_x, train_labels)
    
    fit_time = time.time() - start_time
    
    # valuto probs sul train ma senza cross-validation: è indicativo
    probs = nn.predict_proba(train_x)
    predicts = probs[:,1] # solo le confidenze del neurone (p=sigmoide)
    mse = np.mean((predicts - train_labels)**2)
    
    
    # l'accuracy è discreta. guardando l'mse vedo quanto è convinto il modello
    return nn._convergence_idx, mse, fit_time


def evaluate_lr_tol_convergence(
    max_epoche: int,
    lr: float,
    tol: float,
    train_x: np.ndarray,
    train_labels: np.ndarray
):
    ''' Per analizzare la convergenza al variare del learning rate e la tolleranza '''
    
    
    nn = single_neuron(
        epoche = max_epoche,
        nn_learning_rate = lr,
        activation_function = 'sigmoide',
        convergence_tol=tol
    )    
    
    start_time = time.time()
    
    nn.fit(train_x, train_labels)
    
    fit_time = time.time() - start_time
    
    probs = nn.predict_proba(train_x)
    predicts = probs[:,1] # solo le confidenze del neurone (p=sigmoide)
    mse = np.mean((predicts - train_labels)**2)

    return nn._convergence_idx, mse, fit_time

@njit
def convergence_idx(
    errori,
    tol = 1e-04,
    debug = False
):
    # verifica che sia numpy array
    errori = np.asarray(errori)
    N = errori.shape[0]
    # calcolo somma 10 errori consecutivi, se minore di tol dico che ha raggiunto convergenza
    somma = np.sum(errori[:10])
    
    for i in range(10, N):
        somma -= errori[i-10] # tolgo l'ultimo elemento
        somma += errori[i] # aggiungo il prossimo

        if somma < 10*tol:
            if debug:
                print(f"Converge in {i} step")
            return i
            
    if debug:
        print("Termina per endstep")
    
    return N

def evaluate_convergence_lr(
    lr: float,
    n_epoche: int,
    train_x: np.ndarray,
    train_labels: np.ndarray
):
    
    nn = single_neuron(
        epoche = n_epoche,
        nn_learning_rate = lr,
        activation_function = 'sigmoide',
        return_errors = True,
    )
    
    nn.fit(train_x, train_labels)
    errori = nn.errors_
    
    #if nn.ha
    idx = convergence_idx(
        errori,
        tol = 1e-03,
        debug = True
    )
    
    
    
    return idx