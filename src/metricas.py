"""Metricas compartilhadas. Todo mundo e avaliado exatamente do mesmo jeito."""
import numpy as np
import pandas as pd


def avaliar(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "Erro absoluto medio (MAE, R$)": np.mean(np.abs(y - p)),
        "Raiz do erro quadratico medio (RMSE, R$)": np.sqrt(np.mean((y - p) ** 2)),
        "Erro percentual absoluto medio (MAPE, %)": np.mean(np.abs((y - p) / y)) * 100,
        "Coeficiente de determinacao (R2)": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
    }


def tabela(resultados: dict) -> pd.DataFrame:
    df = pd.DataFrame(resultados).T
    return df.round({
        "Erro absoluto medio (MAE, R$)": 0,
        "Raiz do erro quadratico medio (RMSE, R$)": 0,
        "Erro percentual absoluto medio (MAPE, %)": 2,
        "Coeficiente de determinacao (R2)": 3,
    })
