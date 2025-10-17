import numpy as np
from z3 import *
import time
from typing import Set, Optional

class AbductiveExplanationZ3:

    def __init__(self, rf_model=None):
        self.model = rf_model
        self.solver_calls = 0
        
    def _binarize_features(self, X):

        n_features = X.shape[1] #colunas
        thresholds = [] 
        X_binary = []
        
        for i in range(n_features):
            col = X[:, i]
            threshold = np.median(col)
            thresholds.append(threshold)
            X_binary.append((col > threshold).astype(int)) #compara com a limiar
            
        return np.array(X_binary).T, thresholds
    
    def _create_selector_vars(self, ctx, n_features): 

        return [Bool(f's_{i}', ctx) for i in range(n_features)] #vars das caracteristicas
    
    def _create_coverage_vars(self, ctx, n_refs): #checklist das referencias

        return [Bool(f'p_{i}', ctx) for i in range(n_refs)]
   