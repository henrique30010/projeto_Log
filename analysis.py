import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (7, 4)

#cria uma tabela resumo por dataset e % de referência
def summarize_results(df):
    
    summary = df.groupby(["dataset", "ref_pct"]).agg({
        "successful": "mean",
        "timeouts": "mean",
        "avg_anchors": "mean",
        "avg_time": "mean",
        "avg_solver_calls": "mean",
        "f1_score": "mean"
    }).reset_index()
    print("\n Resumo dos resultados:")
    print(summary)
    return summary

#cria um gráfico da métrica por % de referências para cada dataset
def plot_metric_by_ref(df, metric, title=None):
    sns.lineplot(data=df, x="ref_pct", y=metric, hue="dataset", marker="o")
    plt.title(title or f"{metric} por % de referências")
    plt.xlabel("% de instâncias de referência")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

#cria todos os gráficos principais automaticamente
def plot_dataset_summary(df):
    metrics = ["avg_time", "avg_solver_calls", "avg_anchors", "successful"]
    titles = [
        "Tempo médio de explicação (s)",
        "Número médio de chamadas ao solver",
        "Tamanho médio das explicações (nº âncoras)",
        "Taxa de sucesso (explicações válidas)"
    ]
    for metric, title in zip(metrics, titles):
        plot_metric_by_ref(df, metric, title)

#Mostra correlação entre tempo, solver_calls e âncoras
def correlation_analysis(df):
    corr = df[["avg_time", "avg_solver_calls", "avg_anchors"]].corr()
    print("\nCorrelação entre métricas:")
    print(corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlação entre métricas")
    plt.tight_layout()
    plt.show()

#função para integrar no main
def run_analysis(df):
    
    summary = summarize_results(df)
    plot_dataset_summary(summary)
    correlation_analysis(summary)