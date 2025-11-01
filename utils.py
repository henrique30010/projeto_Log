from itertools import combinations
import numpy as np
from z3 import *


def generate_abductive_reasons_z3(explainer, X_binary, y_pred, x_idx, max_tam):
    n_features = X_binary.shape[1]
    valid_reasons = []
    ctx = Context()

    for i in range(1, max_tam + 1):
        for term_indices in combinations(range(n_features), i):
            if explainer._verify_explanation(ctx, term_indices, X_binary, y_pred, x_idx):
                valid_reasons.append(term_indices)
    return valid_reasons


def filter_minimal_reasons(reasons):
    minimal = []
    for i in reasons:
        if not any(set(other).issubset(i) and set(other) != set(i) for other in reasons):
            minimal.append(i)
    return minimal


def weight(reason, weights):
    return sum(weights[i] for i in reason)
