"""
Abordagens classicas. Duas, de proposito:

  A) Regressao linear so com as colunas tabulares -> o baseline "ingenuo",
     que ignora completamente o texto do anuncio.
  B) Ridge com tabular + TF-IDF do texto -> o baseline JUSTO. Nao adianta
     comparar a LLM com um espantalho: TF-IDF pega palavras como "reformado"
     e "avenida" e recupera boa parte do sinal latente sozinho.

A conclusao do trabalho depende de B existir.
"""
import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = pathlib.Path(__file__).resolve().parents[1]
NUM = ["area", "quartos", "banheiros", "vagas", "idade"]
SEED = 42


def carregar():
    return pd.read_csv(RAIZ / "data" / "anuncios.csv")


def split(df, seed=SEED):
    """Split unico e fixo, usado por TODAS as abordagens. Inegociavel."""
    return train_test_split(df, test_size=0.25, random_state=seed)


def _pre(usar_texto):
    blocos = [
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["bairro"]),
    ]
    if usar_texto:
        blocos.append(
            ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000), "anuncio")
        )
    return ColumnTransformer(blocos)


def modelo_tabular():
    return Pipeline([("pre", _pre(False)), ("reg", LinearRegression())])


def modelo_tabular_tfidf():
    return Pipeline([("pre", _pre(True)), ("reg", Ridge(alpha=1.0))])


def treinar_e_prever(modelo, tr, te):
    # log do alvo: preco e multiplicativo, entao o erro relativo e o que importa
    modelo.fit(tr, np.log(tr.preco))
    return np.exp(modelo.predict(te))


if __name__ == "__main__":
    from metricas import avaliar, tabela

    df = carregar()
    tr, te = split(df)
    linhas = {
        "Regressao linear (so tabular)": treinar_e_prever(modelo_tabular(), tr, te),
        "Ridge (tabular + TF-IDF)": treinar_e_prever(modelo_tabular_tfidf(), tr, te),
    }
    res = {nome: avaliar(te.preco.values, p) for nome, p in linhas.items()}
    print(tabela(res))
    pd.DataFrame({"preco_real": te.preco.values, **linhas}).to_csv(
        RAIZ / "results" / "predicoes_classico.csv", index=False
    )
