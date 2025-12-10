import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from explainer import AbductiveExplanationZ3


def test_single_instance():
    print("="*70)
    print(" TESTE UNITÁRIO: UMA INSTÂNCIA — EXPLICAÇÃO ABDUTIVA Z3")
    print("="*70)

    
    data = load_breast_cancer()
    X, y = data.data, data.target

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    x_idx = 0  # você pode mudar o índice aqui
    print(f"\nInstância selecionada: {x_idx}")

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    
    explainer = AbductiveExplanationZ3(model)
    X_binary, _ = explainer._binarize_features(X_scaled)

    reference_indices = np.random.choice(len(X_binary), 10, replace=False)

    
    print("\n🔍 Gerando explicação...")
    start_time = time.time()
    explanation, k = explainer.find_most_anchored_explanation(
        x_idx, X_binary, y_pred, reference_indices, timeout=20
    )
    elapsed = time.time() - start_time

    print("\n=== RESULTADO DA EXPLICAÇÃO ===")
    if explanation is None:
        print(" Nenhuma explicação encontrada (timeout ou insatisfatível).")
    else:
        print(f" Explicação encontrada em {elapsed:.2f}s")
        print(f"→ Número de âncoras (k): {k}")
        print(f"→ Termo (índices das features): {explanation}")

        # Opcional: mostra nomes das features
        print("\n📋 Features envolvidas na explicação:")
        for i in explanation:
            print(f"  - {i}: {data.feature_names[i]}")

    print("="*70)
    print(f"Total de chamadas ao solver: {explainer.solver_calls}")
    print("="*70)


if __name__ == "__main__":
    test_single_instance()
