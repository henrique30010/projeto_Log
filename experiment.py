import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time
import warnings


from explainer import AbductiveExplanationZ3

warnings.filterwarnings('ignore')

def run_experiments(datasets_config, max_samples=100, output_file="results_z3.csv"):

    results = []
    
    for dataset_name, X, y in datasets_config:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        X = np.array(X, dtype=float)[:max_samples]
        y = np.array(y, dtype=int)[:max_samples]
        
        n_samples, n_features = X.shape
        print(f"Features: {n_features}, Instances: {n_samples}")
        
        # Normaliza features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        explainer = AbductiveExplanationZ3()
        X_binary, _ = explainer._binarize_features(X)
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_idx = 0
        
        for train_idx, test_idx in kf.split(X):
            fold_idx += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            X_train_binary, _ = explainer._binarize_features(X_train)
            X_test_binary, _ = explainer._binarize_features(X_test)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_test_pred = rf.predict(X_test)
            
            f1_test = f1_score(y_test, y_test_pred, average='binary')
            print(f"  Fold {fold_idx}: F1-test={f1_test:.3f}")
            
            # Seleciona referências
            for ref_pct in [5, 10, 20]:
                n_refs = max(1, int(len(X_test) * ref_pct / 100))
                ref_indices = np.random.choice(len(X_test), n_refs, replace=False)
                
                successful = 0
                timeouts = 0
                total_anchors = 0
                times = []
                solver_calls_list = []
                
                n_test_samples = min(5, len(X_test))
                for test_idx_local in range(n_test_samples):
                    explainer.solver_calls = 0
                    t0 = time.time()
                    
                    explanation, k = explainer.find_most_anchored_explanation(
                        test_idx_local, X_test_binary, y_test_pred,
                        ref_indices, timeout=30
                    )
                    
                    elapsed = time.time() - t0
                    
                    if k == "timeout":
                        timeouts += 1
                    elif explanation is not None:
                        successful += 1
                        total_anchors += k
                        times.append(elapsed)
                        solver_calls_list.append(explainer.solver_calls)
                
                avg_time = np.mean(times) if times else 0
                avg_solver_calls = np.mean(solver_calls_list) if solver_calls_list else 0
                
                print(f"    Ref {ref_pct}%: successful={successful}/{n_test_samples}, "
                      f"avg_k={total_anchors/max(1,successful):.1f}, "
                      f"time={avg_time:.2f}s, solver_calls={avg_solver_calls:.0f}")
                
                results.append({
                    'dataset': dataset_name,
                    'fold': fold_idx,
                    'ref_pct': ref_pct,
                    'successful': successful,
                    'timeouts': timeouts,
                    'total_samples': n_test_samples,
                    'avg_anchors': total_anchors / max(1, successful),
                    'avg_time': avg_time,
                    'avg_solver_calls': avg_solver_calls,
                    'f1_score': f1_test
                })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResultados salvos em: {output_file}")
    
    return df_results

