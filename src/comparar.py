"""Roda tudo, gera a tabela final, a curva de aprendizado e a analise de erro."""
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import classico as C
import metricas as M

RAIZ = pathlib.Path(__file__).resolve().parents[1]
(RAIZ / "results").mkdir(exist_ok=True)
import llm as L
TEM_LLM = L.disponivel() or any((RAIZ / "cache").glob("*.json"))


def modelo_hibrido(extras):
    """Mesma regressao linear do baseline + as colunas extraidas pela LLM."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), C.NUM + extras),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["bairro"]),
    ])
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def main():
    df = C.carregar()
    tr, te = C.split(df)
    te_llm = te.head(L.MAX_PRECO_TESTE).copy() if TEM_LLM else te
    tr_llm = tr.head(L.MAX_TREINO_EXTRACAO).copy() if TEM_LLM else tr
    y = te_llm.preco.values
    preds, res = {}, {}

    preds["A. Regressao linear (tabular)"] = C.treinar_e_prever(C.modelo_tabular(), tr, te_llm)
    preds["B. Ridge (tabular + TF-IDF)"] = C.treinar_e_prever(C.modelo_tabular_tfidf(), tr, te_llm)

    if TEM_LLM:
        preds["C. LLM few-shot (estimativa direta)"] = L.prever_precos(tr, te_llm)
        tr2, te2 = tr_llm.copy(), te_llm.copy()
        tr2[L.ATRIBUTOS] = L.extrair_atributos(tr_llm)
        te2[L.ATRIBUTOS] = L.extrair_atributos(te_llm)
        preds["D. Hibrido (LLM extrai -> regressao)"] = C.treinar_e_prever(
            modelo_hibrido(L.ATRIBUTOS), tr2, te2
        )
        print("Uso da API:", L.relatorio_uso())
    else:
        print("!! Ollama fora do ar (servidor local nao respondeu): rodando so as abordagens classicas.\n")
        tr2 = te2 = None

    for nome, p in preds.items():
        res[nome] = M.avaliar(y, p)
    tab = M.tabela(res)
    print(tab.to_string(), "\n")
    tab.to_csv(RAIZ / "results" / "tabela_comparativa.csv")
    pd.DataFrame({"preco_real": y, **preds}).to_csv(RAIZ / "results" / "predicoes.csv", index=False)

    # ---- curva de aprendizado: como cada abordagem se sai com poucos dados
    tamanhos = [20, 40, 80, 160, 320, len(tr)]
    curva = {"A. Regressao linear (tabular)": [], "B. Ridge (tabular + TF-IDF)": []}
    if tr2 is not None:
        curva["D. Hibrido (LLM extrai -> regressao)"] = []
    for n in tamanhos:
        sub = tr.sample(n, random_state=0)
        curva["A. Regressao linear (tabular)"].append(
            M.avaliar(y, C.treinar_e_prever(C.modelo_tabular(), sub, te_llm))["Erro percentual absoluto medio (MAPE, %)"])
        curva["B. Ridge (tabular + TF-IDF)"].append(
            M.avaliar(y, C.treinar_e_prever(C.modelo_tabular_tfidf(), sub, te_llm))["Erro percentual absoluto medio (MAPE, %)"])
        if tr2 is not None:
            n2 = min(n, len(tr2))
            s2 = tr2.sample(n2, random_state=0)
            curva["D. Hibrido (LLM extrai -> regressao)"].append(
                M.avaliar(y, C.treinar_e_prever(modelo_hibrido(L.ATRIBUTOS), s2, te2))["Erro percentual absoluto medio (MAPE, %)"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for nome, vals in curva.items():
        ax.plot(tamanhos, vals, marker="o", label=nome)
    if "C. LLM few-shot (estimativa direta)" in res:
        linha_llm = res["C. LLM few-shot (estimativa direta)"]["Erro percentual absoluto medio (MAPE, %)"]
        ax.axhline(linha_llm, ls="--", color="gray",
                   label="C. LLM few-shot (nao treina)")
    ax.set_xscale("log")
    ax.set_xlabel("Exemplos de treino")
    ax.set_ylabel("MAPE no teste (%) — menor e melhor")
    ax.set_title("Quanto dado cada abordagem precisa?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RAIZ / "results" / "curva_aprendizado.png", dpi=150)
    print("Grafico salvo em results/curva_aprendizado.png")

    # ---- analise de erro: a LLM acertou os atributos escondidos no texto?
    if tr2 is not None:
        gab = pd.read_csv(RAIZ / "data" / "gabarito_latente.csv").loc[te_llm.index]
        acc = {a: (gab[a].astype(int).values == te2[a].values).mean() for a in L.ATRIBUTOS}
        print("\nAcuracia da extracao de atributos latentes:",
              {k: round(v, 3) for k, v in acc.items()})
        pd.Series(acc).to_csv(RAIZ / "results" / "acuracia_extracao.csv")


if __name__ == "__main__":
    main()
