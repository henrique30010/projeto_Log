import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from _experiment import run_experiments
from analysis import run_analysis  

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
    print("resumo dos resultados")
    print("="*70)
    print(results.groupby('dataset').mean())
    
    
    print("\n" + "="*70)
    print("análise dos resultados")
    print("="*70)
    run_analysis(results)
