"""
Duas abordagens com LLM:

  C) LLM few-shot: recebe 12 anuncios do TREINO com preco (para ancorar no
     mercado) e estima o preco dos anuncios de teste direto. Sem few-shot a
     comparacao seria desonesta -- a LLM nao tem como adivinhar o patamar de
     precos deste mercado.

  D) HIBRIDO: a LLM nao estima preco nenhum. Ela so le o anuncio e extrai 4
     atributos booleanos. Esses viram colunas e entram na MESMA regressao
     linear do baseline. E a abordagem que costuma ganhar -- e a moral do
     trabalho: LLM como extrator de features, nao como estimador.

Requer: Ollama rodando localmente (https://ollama.com) + o modelo baixado:
    pip install ollama
    ollama pull llama3.2
Todas as respostas sao cacheadas em cache/ para o resultado ser reproduzivel
sem re-rodar o modelo de novo.
"""
import hashlib
import json
import pathlib
import time

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parents[1]
CACHE = RAIZ / "cache"
CACHE.mkdir(exist_ok=True)
MODELO = "llama3.2"  # modelo local do Ollama; troque por "qwen2.5:3b" (melhor em JSON) se quiser
LOTE = 20  # anuncios por chamada -- reduz o numero de chamadas
MAX_PRECO_TESTE = 20  # usa LLM so nos 20 exemplos de teste mais leves para terminar rapido
MAX_TREINO_EXTRACAO = 60  # extrai atributos só de parte do treino para manter o uso de LLM viavel
CUSTO = {"in": 0.0, "out": 0.0}  # local e gratuito -- contamos tokens so como referencia

_uso = {"in": 0, "out": 0, "chamadas": 0, "segundos": 0.0}


def _chamar(prompt: str) -> str:
    """Chamada com cache em disco pelo hash do prompt."""
    chave = hashlib.sha256((MODELO + prompt).encode()).hexdigest()[:16]
    arq = CACHE / f"{chave}.json"
    if arq.exists():
        return json.loads(arq.read_text())["texto"]

    import ollama

    t0 = time.time()
    r = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        format="json",               # forca saida JSON valida -- ajuda modelos pequenos
        options={"temperature": 0},  # deterministico: mesmo input -> mesma resposta
    )
    _uso["segundos"] += time.time() - t0
    _uso["in"] += r["prompt_eval_count"] or 0
    _uso["out"] += r["eval_count"] or 0
    _uso["chamadas"] += 1
    texto = r["message"]["content"]
    arq.write_text(json.dumps({"prompt": prompt, "texto": texto}))
    return texto


def _json_do_texto(t: str):
    """Extrai a lista de respostas. Modelos pequenos (com format=json) costumam devolver
    um OBJETO no lugar do array pedido -- normalizamos para lista aqui."""
    t = t.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        dado = json.loads(t)
    except json.JSONDecodeError:
        dado = json.loads(t[t.index("[") : t.rindex("]") + 1])  # recorta do 1o [ ao ultimo ]
    if isinstance(dado, dict):
        vals = list(dado.values())
        if len(vals) == 1 and isinstance(vals[0], list):
            return vals[0]  # {"precos": [...]} -> [...]
        return vals  # {"item1": v1, "item2": v2, ...} -> [v1, v2, ...]
    return dado


def _lotes(seq, n=LOTE):
    for i in range(0, len(seq), n):
        yield i, seq[i : i + n]


# ---------------------------------------------------------------- C) few-shot
PROMPT_PRECO = """Voce e um avaliador imobiliario. Abaixo estao anuncios reais deste mercado com o preco pelo qual foram vendidos:

{exemplos}

Agora estime o preco de venda dos anuncios a seguir. Considere que atributos mencionados no texto (estado de conservacao, mobilia, vista, ruido do entorno) tem impacto relevante no preco.

{alvos}

Responda APENAS com um array JSON de numeros, na mesma ordem, sem texto nenhum antes ou depois. Exemplo: [450000, 720000]"""


def _ajusta_numeros(vals, n):
    """Modelos fracos erram a contagem -- forca a resposta a virar exatamente n numeros."""
    nums = []
    for x in vals if isinstance(vals, list) else []:
        try:
            nums.append(float(x))
        except (TypeError, ValueError):
            continue
    nums = nums or [0.0]
    media = sum(nums) / len(nums)
    return (nums + [media] * n)[:n]  # trunca se sobrou, completa com a media se faltou


def prever_precos(tr: pd.DataFrame, te: pd.DataFrame, n_exemplos=12, seed=42):
    ex = tr.sample(n_exemplos, random_state=seed)
    exemplos = "\n\n".join(f"ANUNCIO: {r.anuncio}\nPRECO: R$ {r.preco:,.0f}" for _, r in ex.iterrows())
    te = te.head(MAX_PRECO_TESTE).copy()
    preds = np.zeros(len(te))
    textos = te.anuncio.tolist()
    for i, lote in _lotes(textos):
        alvos = "\n\n".join(f"{j+1}. {t}" for j, t in enumerate(lote))
        vals = _json_do_texto(_chamar(PROMPT_PRECO.format(exemplos=exemplos, alvos=alvos)))
        preds[i : i + len(lote)] = _ajusta_numeros(vals, len(lote))
    return preds


# ------------------------------------------------------------------ D) hibrido
ATRIBUTOS = ["reformado", "mobiliado", "vista_livre", "barulho"]

PROMPT_EXTRACAO = """Leia cada anuncio de imovel e extraia 4 atributos booleanos:

- reformado: o imovel passou por reforma recente?
- mobiliado: vai com moveis/armarios planejados?
- vista_livre: a vista e desimpedida (sem predio na frente)?
- barulho: o entorno e barulhento (avenida movimentada, comercio, eventos)?

ANUNCIOS:
{alvos}

Responda APENAS com um array JSON, um objeto por anuncio na mesma ordem, com exatamente essas 4 chaves e valores true/false. Sem texto antes ou depois."""


def _bool(v):
    if isinstance(v, str):
        return int(v.strip().lower() in ("true", "1", "sim", "yes"))
    return int(bool(v))


def _ajusta_atributos(resp, n):
    """Garante exatamente n registros, cada um com as 4 chaves 0/1 (modelos fracos escorregam)."""
    regs = resp if isinstance(resp, list) else []
    saida = [
        {a: _bool(r.get(a, 0)) for a in ATRIBUTOS} if isinstance(r, dict) else {a: 0 for a in ATRIBUTOS}
        for r in regs[:n]
    ]
    saida += [{a: 0 for a in ATRIBUTOS}] * (n - len(saida))
    return saida


def extrair_atributos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.head(MAX_TREINO_EXTRACAO).copy()
    linhas = []
    for _, lote in _lotes(df.anuncio.tolist()):
        alvos = "\n\n".join(f"{j+1}. {t}" for j, t in enumerate(lote))
        resp = _json_do_texto(_chamar(PROMPT_EXTRACAO.format(alvos=alvos)))
        linhas += _ajusta_atributos(resp, len(lote))
    out = pd.DataFrame(linhas)[ATRIBUTOS].astype(int)
    out.index = df.index
    return out


def relatorio_uso():
    c = _uso["in"] / 1e6 * CUSTO["in"] + _uso["out"] / 1e6 * CUSTO["out"]
    return {**_uso, "custo_usd": round(c, 4)}


def disponivel() -> bool:
    """True se o servidor local do Ollama responde -- usado pelo comparar.py para decidir se roda C e D."""
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False
