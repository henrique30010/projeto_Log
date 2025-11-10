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
    
    def find_kanchored_explanation(self, x_idx, X_binary, y_pred, 
                                reference_indices, k=1, timeout=60):

        ctx = Context()
        set_param('smt.random_seed', 42)
        
        start_time = time.time()
        n_features = X_binary.shape[1]
        
        positive_refs = [i for i in reference_indices if y_pred[i] == y_pred[x_idx]]
        negative_refs = [i for i in reference_indices if y_pred[i] != y_pred[x_idx]]
        
        if len(positive_refs) == 0:
            return None, 0
        
        #  variáveis Z3
        s_vars = self._create_selector_vars(ctx, n_features)
        p_vars = self._create_coverage_vars(ctx, len(positive_refs))
        
        # CNF
        formula = self._build_cnf_formula(ctx, s_vars, p_vars, X_binary, X_binary,
                                            y_pred, x_idx, k, positive_refs, negative_refs)
        
        solver = Solver(ctx=ctx)
        solver.set('timeout', int(timeout * 1000))
        solver.add(formula)
        
        # CEGAR
        iteration = 0
        blocked_terms = []
        
        while time.time() - start_time < timeout:
            iteration += 1
            self.solver_calls += 1
            
            result = solver.check()
            
            if result == unsat:
                if blocked_terms:
                    return None, "refinement"
                else:
                    return None, "unsat"
            
            elif result == unknown:
                return None, "timeout"
            
            else:  # sat
                model = solver.model()
                
                # Extrai o termo do modelo
                term_indices = set()
                for i, s_var in enumerate(s_vars):
                    if model.eval(s_var, model_completion=True):
                        term_indices.add(i)
                
                # Conta referências cobertas
                covered_count = 0
                covers_negative = False
                
                for ref_idx in positive_refs:
                    if all(X_binary[ref_idx, i] == 1 for i in term_indices):
                        covered_count += 1
                
                for ref_idx in negative_refs:
                    if all(X_binary[ref_idx, i] == 1 for i in term_indices):
                        covers_negative = True
                        break
                
                # Verifica se é uma explicação válida
                if not covers_negative and self._verify_explanation(ctx, term_indices, 
                                                                        X_binary, y_pred, x_idx):
                    if covered_count >= k:
                        return sorted(list(term_indices)), covered_count
                
                # Refinamento bloqueia termo 
                blocking_clause = Or([s_vars[i] for i in range(n_features) if i not in term_indices])
                solver.add(blocking_clause)
                blocked_terms.append(term_indices)
        
        return None, "timeout"
    
    def find_most_anchored_explanation(self, x_idx, X_binary, y_pred,
                                       reference_indices, timeout=60):
        
        start_time = time.time()
        best_explanation = None
        best_k = 0
        k = 1
        
        while time.time() - start_time < timeout:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time < 1:
                break
            
            explanation, anchor_count = self.find_kanchored_explanation(
                x_idx, X_binary, y_pred, reference_indices, 
                k=k, timeout=remaining_time
            )
            
            if explanation is None:
                if best_explanation is not None:
                    return best_explanation, best_k
                break
            
            best_explanation = explanation
            best_k = anchor_count
            k = anchor_count + 1
        
        return best_explanation, best_k