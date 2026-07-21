"""
Gera um dataset sintetico de anuncios imobiliarios.

DESENHO EXPERIMENTAL (o coracao do projeto):
O preco depende de 6 atributos ESTRUTURADOS (area, quartos, banheiros, vagas,
idade, bairro) e de 4 atributos LATENTES (reformado, mobiliado, vista_livre,
barulho) que NAO aparecem em nenhuma coluna -- eles so existem, em linguagem
natural e de forma indireta, dentro do texto do anuncio.

Isso cria a pergunta do trabalho: quanto do sinal escondido no texto cada
abordagem consegue recuperar?
"""
import numpy as np
import pandas as pd

SEED = 42
N = 600

BAIRROS = {
    "Centro": 4200, "Sao Mateus": 6800, "Cascatinha": 8100,
    "Santa Luzia": 3500, "Bom Pastor": 5900, "Granbery": 7400,
}

# Efeito de cada atributo latente sobre o preco final (multiplicador)
EFEITO = {"reformado": 1.12, "mobiliado": 1.07, "vista_livre": 1.09, "barulho": 0.88}

FRASES = {
    "reformado": [
        "Recem reformado, tudo novo.",
        "Passou por reforma completa no ano passado.",
        "Cozinha e banheiros reformados ha poucos meses.",
    ],
    "nao_reformado": [
        "Imovel no estado original, precisa de uma atualizacao.",
        "Acabamento da epoca da construcao, otimo para quem quer personalizar.",
        "Conservado, porem sem reformas recentes.",
    ],
    "mobiliado": [
        "Vai mobiliado, com armarios planejados em todos os comodos.",
        "Entrega com moveis planejados e eletrodomesticos.",
    ],
    "nao_mobiliado": [
        "Entregue sem moveis.",
        "Sem armarios, pronto para o proprietario montar do seu jeito.",
    ],
    "vista_livre": [
        "Vista livre e desimpedida para a serra.",
        "Andar alto, sem predios na frente, muito sol pela manha.",
        "A vista da sala e um espetaculo ao entardecer.",
    ],
    "nao_vista_livre": [
        "As janelas dao para o predio vizinho.",
        "Vista para os fundos do quarteirao.",
    ],
    "barulho": [
        "Fica de frente para uma avenida de trafego intenso.",
        "Proximo a uma casa de eventos, regiao movimentada a noite.",
        "Fica sobre um comercio bem movimentado.",
    ],
    "nao_barulho": [
        "Rua tranquila e residencial, sem movimento.",
        "Localizacao silenciosa, otima para quem trabalha em casa.",
    ],
}


def _texto(r, rng):
    p = [
        f"{'Apartamento' if r.vagas <= 2 else 'Casa'} de {r.area:.0f}m2 no bairro {r.bairro}, "
        f"com {r.quartos} quartos e {r.banheiros} banheiros.",
        f"{'Sem vaga de garagem.' if r.vagas == 0 else f'{r.vagas} vaga(s) de garagem.'}",
        f"Construcao de aproximadamente {r.idade} anos.",
    ]
    for attr in ["reformado", "mobiliado", "vista_livre", "barulho"]:
        chave = attr if r[attr] else f"nao_{attr}"
        p.append(rng.choice(FRASES[chave]))
    rng.shuffle(p[1:])
    return " ".join(p)


def gerar(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area": rng.normal(95, 35, n).clip(30, 320).round(0),
        "quartos": rng.integers(1, 5, n),
        "banheiros": rng.integers(1, 4, n),
        "vagas": rng.integers(0, 4, n),
        "idade": rng.integers(0, 45, n),
        "bairro": rng.choice(list(BAIRROS), n),
    })
    for attr in EFEITO:
        df[attr] = rng.random(n) < (0.25 if attr == "barulho" else 0.45)

    base = (
        df.bairro.map(BAIRROS) * df.area
        + df.quartos * 9000
        + df.banheiros * 12000
        + df.vagas * 18000
        - df.idade * 2200
    )
    for attr, mult in EFEITO.items():
        base = base * np.where(df[attr], mult, 1.0)
    ruido = rng.normal(1.0, 0.06, n)  # ruido irredutivel: nenhum modelo passa disso
    df["preco"] = (base * ruido).round(-3)

    rng_txt = np.random.default_rng(seed + 1)
    df["anuncio"] = [_texto(r, rng_txt) for _, r in df.iterrows()]

    # Os atributos latentes ficam guardados so como gabarito da analise de erro.
    gabarito = df[list(EFEITO)].copy()
    df = df.drop(columns=list(EFEITO))
    return df, gabarito


if __name__ == "__main__":
    import pathlib

    df, gabarito = gerar()
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "anuncios.csv", index=False)
    gabarito.to_csv(out / "gabarito_latente.csv", index=False)
    print(df[["area", "quartos", "bairro", "preco"]].describe(include="all").T)
    print("\nExemplo de anuncio:\n", df.anuncio.iloc[0])
