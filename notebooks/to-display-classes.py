


from sklearn.base import ClassifierMixin, BaseEstimator

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
        
        
    