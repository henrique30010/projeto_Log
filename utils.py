# utils

from itertools import combinations
import numpy as np


def generate_abductive_reasons(explainer, X_binary, y_pred, x_idx, max_tam):
    n_features = X_binary.shape[1]
    valid_reasons = []
    for i in range(1, max_tam + 1):
        for term_indices in combinations(range(n_features), i):
            if explainer.is_abductive_explanation(term_indices, x_idx, X_binary, y_pred):
                valid_reasons.append(term_indices)
        return valid_reasons


def filter_minimal_reasons(reasons):
    minimal = []
    for i in reasons:
        if not any(set(other).issubset(i) and set(other) != set(i) for other in reasons):
            minimal.append(i)
    return minimal

# calcula o peso de uma razão // menor peso total = razão preferida


def weight(reasons, weights):
    return sum(weights[i] for i in reason)
