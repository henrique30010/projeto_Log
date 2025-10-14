import numpy as np
import time
from itertools import combinations
from typing import List, Optional, Set, Tuple

class AbductiveExplanation:

        def __init__(self, rf_model, domain_theory=None):

            self.model = rf_model
            self.domain_theory = domain_theory
            
        def _get_binary_features(self, X):

            n_features = X.shape[1] #colunas
            binary_features = []
            thresholds = []
            
            for i in range(n_features):
                col = X[:, i]
                
                threshold = np.median(col)
                thresholds.append(threshold)
                binary_features.append(col > threshold)
                
            return np.array(binary_features).T, thresholds
        
        def _instance_satisfies_term(self, instance_binary, term_indices):

            return all(instance_binary[i] for i in term_indices)
        
        def _is_abductive_explanation(self, term_indices, x_idx, X_binary, y_pred):
            
            # x_binary base de dados e y_prend lista das previsoes 
            y_target = y_pred[x_idx]
            x_binary = X_binary[x_idx] 
            
            for idx in range(len(X_binary)):
                if self._instance_satisfies_term(X_binary[idx], term_indices):
                    if y_pred[idx] != y_target:
                        return False
            return True