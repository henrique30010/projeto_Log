import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

from _experiment import run_experiments

if __name__ == "__main__":
    print("Carregando datasets...")
    
    breast_cancer = load_breast_cancer()
    wine = load_wine()
    
    datasets_config = [
        ("breast_cancer", breast_cancer.data, breast_cancer.target),
        ("wine", wine.data, (wine.target == 0).astype(int)),
    ]
    
    results = run_experiments(datasets_config, max_samples=150)
    
    print("\n" + "="*70)
    print("RESUMO DOS RESULTADOS")
    print("="*70)
    summary = results.groupby('dataset').agg({
        'successful': 'mean',
        'timeouts': 'mean',
        'avg_anchors': 'mean',
        'avg_time': 'mean',
        'avg_solver_calls': 'mean'
    })
    print(summary)