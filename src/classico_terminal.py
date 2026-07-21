"""Roda apenas as abordagens classicas e imprime uma tabela no terminal.

Mostra, para cada anuncio do teste, o valor real, a previsao de cada modelo
classico e o erro absoluto.
"""
import numpy as np
import pandas as pd

import classico as C
from metricas import avaliar, tabela


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)

    df = C.carregar()
    tr, te = C.split(df)

    pred_a = C.treinar_e_prever(C.modelo_tabular(), tr, te)
    pred_b = C.treinar_e_prever(C.modelo_tabular_tfidf(), tr, te)

    resumo = {
        "Regressao linear (so tabular)": avaliar(te.preco.values, pred_a),
        "Regressao linear com TF-IDF": avaliar(te.preco.values, pred_b),
    }

    tabela_resumo = tabela(resumo)
    print("\n=== Resumo das metricas ===\n")
    print(tabela_resumo.to_string())

    detalhes = pd.DataFrame(
        {
            "preco_real": te.preco.values,
            "pred_tabular": pred_a,
            "erro_tabular": pred_a - te.preco.values,
            "abs_erro_tabular": np.abs(pred_a - te.preco.values),
            "pred_reg_tfidf": pred_b,
            "erro_reg_tfidf": pred_b - te.preco.values,
            "abs_erro_reg_tfidf": np.abs(pred_b - te.preco.values),
        }
    )

    detalhes = detalhes.sort_values("abs_erro_reg_tfidf", ascending=False)

    print("\n=== Tabela no terminal: valor real x previsoes ===\n")
    print(
        detalhes.round(
            {
                "preco_real": 0,
                "pred_tabular": 0,
                "erro_tabular": 0,
                "abs_erro_tabular": 0,
                "pred_reg_tfidf": 0,
                "erro_reg_tfidf": 0,
                "abs_erro_reg_tfidf": 0,
            }
        ).to_string()
    )


if __name__ == "__main__":
    main()