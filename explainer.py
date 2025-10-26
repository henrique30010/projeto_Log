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

    def _build_cnf_formula(self, ctx, s_vars, p_vars, x_binary, ref_binary, 
                        y_pred, x_idx, a, positive_refs, negative_refs):
        # formulas
        constraints = []
        n_features = len(s_vars)
        
        # 1 Restrições de compatibilidade
        for p_idx, ref_idx in enumerate(positive_refs):
            for lit_idx in range(n_features):
            
                if x_binary[x_idx, lit_idx] == 1 and ref_binary[ref_idx, lit_idx] == 0:
                    constraints.append(Or(Not(s_vars[lit_idx]), Not(p_vars[p_idx])))
        
        # 2 Cobertura mínima 
        coverage_sum = Sum([If(p_vars[i], 1, 0) for i in range(len(p_vars))])
        constraints.append(coverage_sum >= a)
        
        # 3 Não cobrir referências com classe diferente
        for neg_ref_idx in negative_refs:
            clause = []
            for lit_idx in range(n_features):
                if x_binary[x_idx, lit_idx] == 1 and ref_binary[neg_ref_idx, lit_idx] == 0:
                    clause.append(s_vars[lit_idx])
            if clause:
                constraints.append(Or(clause))
        
        # Restrições de cobertura de referências
        for p_idx, ref_idx in enumerate(positive_refs):
            all_match = []
            for lit_idx in range(n_features):
                if x_binary[x_idx, lit_idx] == 1:
                    if ref_binary[ref_idx, lit_idx] == 1:
                        all_match.append(s_vars[lit_idx])
            
            if all_match:
                constraints.append(Implies(p_vars[p_idx], And(all_match)))
        
        return And(constraints) if constraints else BoolVal(True, ctx)
    
    def _verify_explanation(self, ctx, term_indices, x_binary, y_pred, x_idx):

        solver = Solver(ctx=ctx)
        self.solver_calls += 1
        
        target_class = y_pred[x_idx]
        
        for idx in range(len(x_binary)):
            covered = all(x_binary[idx, lit] == 1 for lit in term_indices)
            if covered and y_pred[idx] != target_class:
                return False
        
        return True