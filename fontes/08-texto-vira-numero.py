# %% [markdown]
# # Módulo 08 — Linguagem 1: texto vira número
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/08-texto-vira-numero.ipynb)
#
# **Objetivo.** Dar o primeiro passo rumo aos modelos de linguagem, como o ChatGPT e o Claude: transformar texto em números e
# prever o próximo caractere. Sem redes neurais ainda: só contagem e probabilidade.
#
# **O que você vai construir.** Um modelo de bigramas que lê *Alice no País das Maravilhas*, em inglês, e escreve um "inglês"
# novo, letra por letra. Você também vai medir a qualidade do modelo com a perplexidade.

# %% [markdown]
# ## 1. A intuição: adivinhar a próxima letra
#
# Quando você digita no celular, o teclado sugere a próxima palavra. Ele aprendeu, com muito texto, o que costuma vir depois
# do que você já escreveu. Um modelo de linguagem (language model) faz exatamente isso: dado um trecho de texto, dá a
# **probabilidade** de cada continuação possível.
#
# Faça o teste: em inglês, depois da letra **q**, qual vem? Quase sempre **u** (*queen*, *quite*, *question*). E depois de
# **th**? Muitas vezes **e** (*the*, *then*, *there*). Você aprendeu essas regularidades lendo. O computador vai aprendê-las
# contando.
#
# Os grandes modelos de linguagem são, no fundo, máquinas gigantes de "prever o próximo pedaço de texto". Neste módulo
# construímos a versão mais simples possível. Nos Módulos 09 e 10, vamos melhorá-la até chegar a um Transformer em miniatura.

# %% [markdown]
# ## 2. O texto
#
# Usamos *Alice's Adventures in Wonderland*, de Lewis Carroll (1865), em domínio público. O arquivo vem do Projeto Gutenberg,
# na seleção distribuída pela biblioteca NLTK. Uma cópia fica na pasta `dados/` do repositório. Como o Colab abre só o notebook, a função
# abaixo procura o arquivo localmente e, se não achar, pega o mesmo texto no corpus Gutenberg da NLTK, que baixa os livros de um repositório
# público. Se a NLTK não estiver instalada, a função a instala antes (`subprocess` e `sys` servem só para rodar o `pip`, o instalador de
# pacotes do Python).

# %%
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

CAMINHO_LOCAL = "../dados/alice.txt"


def carregar_texto():
    """Lê o livro da pasta dados/ ou, se não existir (no Colab), pega o mesmo texto no corpus Gutenberg da NLTK."""
    if os.path.exists(CAMINHO_LOCAL):
        with open(CAMINHO_LOCAL, encoding="utf-8") as arquivo:
            return arquivo.read()
    try:
        import nltk
    except ImportError:                                   # instala a NLTK, se ainda não estiver instalada
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=True)
        import nltk
    nltk.download("gutenberg", quiet=True, raise_on_error=True)   # baixa os livros (só na primeira vez)
    return nltk.corpus.gutenberg.raw("carroll-alice.txt")


texto = carregar_texto().lower()   # tudo em minúsculas, para simplificar
print(f"{len(texto)} caracteres")
print(texto[:500])

# %% [markdown]
# Passamos tudo para minúsculas: "The" e "the" viram a mesma coisa. Isso diminui o número de símbolos diferentes.

# %% [markdown]
# ## 3. Tokens e vocabulário
#
# Para um modelo, o texto é uma sequência de unidades chamadas tokens (tokens). Há várias escolhas:
#
# - **caracteres:** cada letra, espaço ou pontuação é um token. Poucos tokens diferentes, mas cada um carrega pouco significado;
# - **palavras:** cada palavra é um token. Mais significado, mas muitos tokens diferentes, e palavras novas ficam de fora;
# - **pedaços de palavras (subwords):** o meio-termo usado pelos grandes modelos. Por exemplo, *unbelievable* pode virar
#   *un* + *believ* + *able*.
#
# O conjunto de todos os tokens diferentes é o vocabulário (vocabulary). Neste módulo usamos caracteres. Primeiro, uma espiada
# no texto como palavras, que também é um bom exercício de inglês. `re.findall` acha todos os trechos que seguem um padrão; aqui,
# sequências de letras e apóstrofos. `Counter` conta quantas vezes cada coisa aparece.

# %%
palavras = re.findall(r"[a-z']+", texto)
contagem_palavras = Counter(palavras)
print(f"{len(palavras)} palavras, {len(contagem_palavras)} palavras diferentes")
print("as 15 mais comuns:", contagem_palavras.most_common(15))

# %% [markdown]
# Agora o vocabulário de caracteres. Cada caractere ganha um número (seu índice). Codificar (encode) é trocar texto por
# números; decodificar (decode) é o caminho inverso. É assim, literalmente, que "texto vira número".

# %%
vocabulario = sorted(set(texto))
tamanho_vocabulario = len(vocabulario)
indice_de = {caractere: i for i, caractere in enumerate(vocabulario)}    # caractere -> número
caractere_de = {i: caractere for i, caractere in enumerate(vocabulario)}  # número -> caractere

print(f"{tamanho_vocabulario} caracteres diferentes:")
print(repr("".join(vocabulario)))


def codificar(trecho):
    """Texto -> array de índices."""
    return np.array([indice_de[c] for c in trecho])


def decodificar(indices):
    """Array de índices -> texto."""
    return "".join(caractere_de[int(i)] for i in indices)


print(codificar("alice"))
print(decodificar(codificar("alice")))

# %% [markdown]
# (O `'\n'` no vocabulário é a quebra de linha. Um dicionário criado com `{chave: valor for ...}` é como uma lista criada com
# `[... for ...]`, só que de pares.)

# %% [markdown]
# ## 4. Probabilidade condicional
#
# A probabilidade de um caractere depende do que veio antes. "Qual a chance de a próxima letra ser **u**?" é uma pergunta.
# "Qual a chance de a próxima letra ser **u**, **sabendo que** a atual é **q**?" é outra, com resposta muito diferente.
#
# A segunda é uma probabilidade condicional (conditional probability), escrita $P(\text{u} \mid \text{q})$; a barra se lê "dado que".
# Contando, ela é:
#
# $$ P(b \mid a) = \frac{\text{número de vezes que } b \text{ vem logo depois de } a}{\text{número de vezes que } a \text{ aparece seguido de alguma coisa}} $$
#
# **Exemplo à mão**, com o texto "the cat sat on the mat". Os caracteres que vêm depois de cada **t**:
#
# - "**t**he" → h; "ca**t** " → espaço; "sa**t** " → espaço; "**t**he" → h; o **t** final de "mat" não tem sucessor.
#
# São 4 casos: 2 vezes **h** e 2 vezes espaço. Logo $P(\text{h} \mid \text{t}) = 2/4 = 0{,}5$ e $P(\text{espaço} \mid \text{t}) = 0{,}5$.
# Depois de **a** ("c**a**t", "s**a**t", "m**a**t") vem sempre **t**: $P(\text{t} \mid \text{a}) = 3/3 = 1$.

# %% [markdown]
# ## 5. O modelo de bigramas
#
# Um bigrama (bigram) é um par de tokens seguidos. O modelo de bigramas faz a suposição mais simples possível: o próximo caractere
# depende **só** do caractere atual. Ele é treinado contando todos os pares do texto numa tabela: a linha é o caractere atual,
# a coluna é o próximo.
#
# Antes de contar, separamos o texto: os primeiros 90% para treino e os últimos 10% para validação (Módulo 05). Aqui não sorteamos
# as posições, como fizemos com as imagens, porque o texto é uma sequência: sortear caracteres soltos destruiria os pares.

# %%
indices_texto = codificar(texto)
corte = int(0.9 * len(indices_texto))
indices_treino, indices_validacao = indices_texto[:corte], indices_texto[corte:]
print(f"treino: {len(indices_treino)} caracteres   validação: {len(indices_validacao)} caracteres")


def contar_bigramas(indices):
    """Tabela de contagens: linha = caractere atual, coluna = caractere seguinte."""
    contagens = np.zeros((tamanho_vocabulario, tamanho_vocabulario))
    for atual, seguinte in zip(indices[:-1], indices[1:]):
        contagens[atual, seguinte] += 1
    return contagens


contagens = contar_bigramas(indices_treino)
print(f"pares contados: {int(contagens.sum())}")
print(f"'q' foi seguido de 'u' {int(contagens[indice_de['q'], indice_de['u']])} vezes, "
      f"de qualquer caractere {int(contagens[indice_de['q']].sum())} vezes")

# %% [markdown]
# Um mapa de calor (heatmap) das contagens, só para o espaço e as letras de a a z. Cada quadrado é um par; quanto mais escuro,
# mais frequente. Usamos escala logarítmica de cor, porque alguns pares são milhares de vezes mais comuns que outros.

# %%
from matplotlib.colors import LogNorm

letras = [" "] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
posicoes = [indice_de[c] for c in letras]
sub_tabela = contagens[np.ix_(posicoes, posicoes)] + 1   # +1 só para o log não ver zeros

fig, eixo = plt.subplots(figsize=(8, 7))
imagem = eixo.imshow(sub_tabela, cmap="Blues", norm=LogNorm())
rotulos_eixo = ["␣" if c == " " else c for c in letras]
eixo.set_xticks(range(len(letras)), rotulos_eixo)
eixo.set_yticks(range(len(letras)), rotulos_eixo)
eixo.set_xlabel("caractere seguinte")
eixo.set_ylabel("caractere atual")
eixo.set_title("Contagem de bigramas em Alice (␣ = espaço)")
fig.colorbar(imagem, label="contagem + 1")
plt.show()

# %% [markdown]
# Dá para ver o inglês na tabela: a linha do **q** quase só tem a coluna **u**; a linha do **t** é forte na coluna **h**;
# quase toda letra é seguida com frequência por espaço (fim de palavra). (`np.ix_` seleciona as linhas e colunas escolhidas.)
#
# ### De contagens a probabilidades
#
# Dividimos cada linha pelo seu total. Mas há um problema: um par que nunca apareceu no treino receberia probabilidade **zero**.
# Se ele aparecer na validação, o modelo diria "impossível", e o $\log 0$ da perda seria infinito. A solução mais simples é a
# suavização (smoothing): somar 1 a todas as contagens antes de dividir. Nenhum par fica impossível, só improvável.

# %%
def probabilidades_bigrama(contagens, suavizacao=1.0):
    """P(seguinte | atual): cada linha da tabela de contagens (mais a suavização), dividida pelo seu total."""
    suavizadas = contagens + suavizacao
    return suavizadas / suavizadas.sum(axis=1, keepdims=True)


probabilidades = probabilidades_bigrama(contagens)
print(f"P('u' | 'q') = {probabilidades[indice_de['q'], indice_de['u']]:.3f}")
print(f"P('h' | 't') = {probabilidades[indice_de['t'], indice_de['h']]:.3f}")
print(f"P('e' | 'h') = {probabilidades[indice_de['h'], indice_de['e']]:.3f}")

# %% [markdown]
# Repare no preço da suavização: sem ela, $P(\text{u} \mid \text{q})$ seria $185/186 \approx 0{,}99$; com ela, caiu para cerca de 0,8.
# O **q** aparece só 186 vezes, e somar 1 em cada uma das 47 colunas pesa bastante. Letras comuns quase não são afetadas.
# Por isso, às vezes usa-se uma suavização menor que 1 (o Exercício 3 usa 0,1).

# %% [markdown]
# ## 6. Gerando texto por amostragem
#
# Para escrever, o modelo sorteia o próximo caractere de acordo com as probabilidades da linha do caractere atual. Depois o
# sorteado vira o atual, e assim por diante. Isso é amostragem (sampling). Sortear, e não pegar sempre o mais provável, é o que
# faz o texto variar: se escolhêssemos sempre o mais provável, o modelo repetiria a mesma coisa para sempre.
#
# `gerador.choice(n, p=...)` sorteia um número de 0 a $n-1$, com as probabilidades dadas. O parâmetro `temperatura` será explicado
# na seção 7; por enquanto, vale 1.

# %%
def gerar_texto(probabilidades, inicio, n_caracteres, temperatura=1.0, semente=0):
    """Escreve n_caracteres sorteando cada um a partir do anterior."""
    gerador = np.random.default_rng(semente)
    atual = indice_de[inicio]
    saida = [inicio]
    for _ in range(n_caracteres):
        p = probabilidades[atual] ** (1 / temperatura)
        p = p / p.sum()
        atual = gerador.choice(tamanho_vocabulario, p=p)
        saida.append(caractere_de[atual])
    return "".join(saida)


print(gerar_texto(probabilidades, "a", 400))

# %% [markdown]
# Não é inglês, mas também não é qualquer coisa. Há espaços nos lugares certos, palavras curtas reais (*the*, *and*, *she*) e
# pedaços que "soam" ingleses. Compare com um texto sem modelo nenhum, todos os caracteres igualmente prováveis:

# %%
uniforme = np.full((tamanho_vocabulario, tamanho_vocabulario), 1 / tamanho_vocabulario)
print(gerar_texto(uniforme, "a", 200))

# %% [markdown]
# ## 7. Temperatura: mais cautela ou mais ousadia
#
# A temperatura (temperature) $T$ ajusta o sorteio. Elevamos cada probabilidade a $1/T$ e normalizamos de novo:
#
# $$ p_i^{(T)} = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}} $$
#
# - $T = 1$: as probabilidades do modelo, sem mudança;
# - $T < 1$: as maiores ficam ainda maiores; o texto fica mais previsível e repetitivo;
# - $T > 1$: as probabilidades se aproximam umas das outras; o texto fica mais variado e mais estranho.
#
# **Exemplo à mão**, com três opções de probabilidade $(0{,}6;\ 0{,}3;\ 0{,}1)$:
#
# - $T = 0{,}5$ (elevar ao quadrado): $0{,}36;\ 0{,}09;\ 0{,}01$, soma $0{,}46$, resultado $\approx (0{,}783;\ 0{,}196;\ 0{,}022)$;
# - $T = 2$ (tirar a raiz quadrada): $0{,}775;\ 0{,}548;\ 0{,}316$, soma $1{,}639$, resultado $\approx (0{,}473;\ 0{,}334;\ 0{,}193)$.
#
# **Curiosidade de Física.** O nome não é por acaso. Na distribuição de Boltzmann, a chance de um sistema estar num estado de energia
# $E$ é proporcional a $e^{-E/kT}$. Em temperatura baixa, quase tudo fica no estado de menor energia; em temperatura alta, os estados
# se misturam. A temperatura dos modelos de linguagem faz o mesmo com as escolhas do próximo token.

# %%
p_exemplo = np.array([0.6, 0.3, 0.1])
for T in [0.5, 1.0, 2.0]:
    ajustada = p_exemplo ** (1 / T)
    print(f"T = {T}: {np.round(ajustada / ajustada.sum(), 3)}")

# %% [markdown]
# O gráfico mostra o efeito nas probabilidades do caractere que vem depois de **t**, para as letras mais comuns.

# %%
linha_t = probabilidades[indice_de["t"]]
mais_comuns = np.argsort(linha_t)[::-1][:8]
fig, eixo = plt.subplots(figsize=(9, 4))
largura = 0.25
for deslocamento, T in zip([-largura, 0, largura], [0.5, 1.0, 2.0]):
    ajustada = linha_t ** (1 / T)
    ajustada = ajustada / ajustada.sum()
    eixo.bar(np.arange(8) + deslocamento, ajustada[mais_comuns], width=largura, label=f"T = {T}")
eixo.set_xticks(range(8), [repr(caractere_de[i]) for i in mais_comuns])
eixo.set_xlabel("caractere seguinte")
eixo.set_ylabel("probabilidade")
eixo.set_title("Depois de 't': o efeito da temperatura")
eixo.legend()
eixo.grid(True, axis="y")
plt.show()

# %%
for T in [0.5, 1.0, 2.0]:
    print(f"--- temperatura {T} ---")
    print(gerar_texto(probabilidades, "t", 250, temperatura=T, semente=1))
    print()

# %% [markdown]
# Com $T = 0{,}5$ aparecem muitos "the" e "he" repetidos; com $T = 2$, quase uma sopa de letras. Os assistentes de IA também têm
# esse controle: temperatura baixa para respostas mais estáveis, mais alta para textos mais criativos.

# %% [markdown]
# ## 8. Medindo o modelo: a perplexidade
#
# Como saber se um modelo de linguagem é bom? Olhamos o texto de validação, que ele nunca viu, e perguntamos: que probabilidade o
# modelo dava para cada caractere que **realmente** veio? Quanto maiores essas probabilidades, melhor.
#
# A medida natural é a entropia cruzada do Módulo 05, aqui calculada sobre cada caractere da validação:
#
# $$ L = -\frac{1}{n} \sum_{i} \log P(c_{i+1} \mid c_i) $$
#
# A perplexidade (perplexity) é $e^{L}$. Ela tem uma interpretação intuitiva: **o modelo está tão indeciso quanto alguém que
# escolhesse entre $e^L$ opções igualmente prováveis.**
#
# **Exemplo à mão.** Se o modelo desse sempre probabilidade $1/4$ ao caractere certo, $L = -\log(1/4) = \log 4$, e a perplexidade seria
# $e^{\log 4} = 4$: como escolher entre 4 opções. Um modelo que chuta ao acaso entre os 47 caracteres tem perplexidade 47.
# Quanto menor, melhor; o mínimo, para um modelo que acerta tudo com certeza, é 1.

# %%
def perplexidade_bigrama(probabilidades, indices):
    """e elevado à entropia cruzada média, sobre todos os pares do texto."""
    p_corretos = probabilidades[indices[:-1], indices[1:]]
    return np.exp(-np.mean(np.log(p_corretos)))


# Modelo "unigrama": só a frequência de cada caractere, sem olhar o anterior
frequencias = np.bincount(indices_treino, minlength=tamanho_vocabulario) + 1.0
frequencias = frequencias / frequencias.sum()
perplexidade_unigrama = np.exp(-np.mean(np.log(frequencias[indices_validacao[1:]])))

print(f"chute uniforme:                  perplexidade {tamanho_vocabulario}")
print(f"unigrama (só frequência):        perplexidade {perplexidade_unigrama:.1f}")
print(f"bigrama (olha 1 caractere atrás): perplexidade {perplexidade_bigrama(probabilidades, indices_validacao):.1f}")

# %% [markdown]
# Cada degrau de "memória" diminui a indecisão. O bigrama hesita, em média, como se escolhesse entre uns 10 a 12 caracteres.
# (`np.bincount` conta quantas vezes aparece cada número.)
#
# **O aprendizado acontecendo.** Aqui, "aprender" é contar. Então, quanto mais texto o modelo lê, melhores as contagens. Vamos treinar
# com pedaços cada vez maiores do treino e medir a perplexidade na validação.

# %%
fracoes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
perplexidades = []
for fracao in fracoes:
    pedaco = indices_treino[:int(fracao * len(indices_treino))]
    perplexidades.append(perplexidade_bigrama(probabilidades_bigrama(contar_bigramas(pedaco)), indices_validacao))

caracteres_lidos = [int(f * len(indices_treino)) for f in fracoes]
plt.figure(figsize=(7, 4))
plt.plot(caracteres_lidos, perplexidades, marker="o")
plt.xscale("log")
plt.xlabel("caracteres lidos no treino (escala logarítmica)")
plt.ylabel("perplexidade na validação")
plt.title("Quanto mais o modelo lê, menos perplexo ele fica")
plt.grid(True)
plt.show()

# %% [markdown]
# Com poucas centenas de caracteres, o modelo mal viu cada par. A perplexidade cai depressa no começo e depois se estabiliza:
# com um caractere só de memória, ler mais não ajuda muito. Para melhorar de verdade, o modelo precisa olhar **mais para trás**.

# %% [markdown]
# ## 9. O limite da contagem
#
# Por que não contar trios, quádruplas, sequências de 10 caracteres? O número de combinações explode:
#
# | contexto | combinações possíveis ($47^{\text{contexto}}$) |
# |---|---|
# | 1 caractere | 47 |
# | 2 caracteres | 2 209 |
# | 3 caracteres | 103 823 |
# | 5 caracteres | cerca de 229 milhões |
#
# O livro inteiro tem só 144 mil caracteres. Com contextos longos, quase todas as combinações nunca aparecem, e a tabela fica cheia
# de zeros. O modelo não consegue **generalizar**: não percebe que "the cat" e "the dog" têm algo em comum.
#
# É aqui que entram as redes neurais. No próximo módulo, cada caractere vira um vetor de números (um *embedding*), e uma rede aprende
# a combinar vários caracteres anteriores sem precisar de uma tabela gigante.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Para o texto "the cat sat on the mat", calcule à mão $P(\text{h} \mid \text{t})$, $P(\text{t} \mid \text{a})$
# e $P(\text{a} \mid \text{c})$. Confira contando os pares com `zip(frase[:-1], frase[1:])` e um `Counter`.
#
# **Exercício 2 (médio).** Faça um modelo de bigramas de **palavras**: para cada palavra, conte quais palavras vêm logo depois
# (um dicionário de `Counter`s). Gere 40 palavras começando por "alice", sorteando cada próxima palavra proporcionalmente às contagens.
# Compare com o texto de caracteres. Quantas palavras diferentes há? Por que a tabela de palavras teria quase só zeros?
#
# **Exercício 3 (desafio).** Faça um modelo de **trigramas** de caracteres: o próximo depende dos **dois** anteriores. Use um array de
# forma $(47, 47, 47)$, suavização 0,1 e meça a perplexidade na validação. Ela melhora em relação ao bigrama? Gere 300 caracteres a
# partir de "th". (Dica: `np.add.at(contagens, (a, b, c), 1)` soma 1 em todas as posições de uma vez.)

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | language model | modelo de linguagem | *A language model predicts the next token.* |
# | token | token | *Here, each character is a token.* |
# | vocabulary | vocabulário | *Our vocabulary has 47 characters.* |
# | corpus | corpus | *The corpus is Alice in Wonderland.* |
# | encode / decode | codificar / decodificar | *Encode the text into numbers, then decode it back.* |
# | bigram | bigrama | *"th" is a very common bigram in English.* |
# | conditional probability | probabilidade condicional | *The conditional probability of "u" given "q" is almost one.* |
# | frequency | frequência | *"e" has the highest frequency among letters.* |
# | smoothing | suavização | *Smoothing avoids zero probabilities.* |
# | sampling | amostragem | *Sampling makes every generated text different.* |
# | temperature | temperatura | *A low temperature makes the text repetitive.* |
# | perplexity | perplexidade | *Lower perplexity means a better model.* |
# | context | contexto | *A bigram model has a context of one character.* |
# | heatmap | mapa de calor | *The heatmap shows which pairs are common.* |

# %% [markdown]
# ## Resumo
#
# - Para o computador, texto é uma sequência de tokens (aqui, caracteres), cada um trocado por um número do vocabulário.
# - Um modelo de linguagem dá a probabilidade do próximo token; o de bigramas usa só o token atual, $P(b \mid a)$, obtida contando pares.
# - Para gerar texto, sorteamos o próximo token; a temperatura deixa o sorteio mais cauteloso ($T < 1$) ou mais ousado ($T > 1$).
# - A perplexidade, $e^{\text{entropia cruzada}}$, mede de quantas opções o modelo está "indeciso"; quanto menor, melhor.
# - Contar não escala para contextos longos: as combinações explodem. É a deixa para as redes neurais.
#
# **Próximo módulo:** um mini modelo neural que olha vários caracteres para trás e escreve um inglês mais convincente.

# %% [markdown]
# ## Soluções
#
# **Aviso:** só olhe depois de tentar de verdade. Errar faz parte do aprendizado.
#
# &nbsp;
#
# &nbsp;
#
# &nbsp;

# %% [markdown]
# ### Solução do Exercício 1
#
# - Depois de **t**: h, espaço, espaço, h. $P(\text{h} \mid \text{t}) = 2/4 = 0{,}5$.
# - Depois de **a**: t, t, t. $P(\text{t} \mid \text{a}) = 3/3 = 1$.
# - Depois de **c**: a (só em "cat"). $P(\text{a} \mid \text{c}) = 1/1 = 1$.

# %%
frase = "the cat sat on the mat"
pares = Counter(zip(frase[:-1], frase[1:]))
for atual, seguinte in [("t", "h"), ("a", "t"), ("c", "a")]:
    total_atual = sum(n for (a, _), n in pares.items() if a == atual)
    print(f"P({seguinte!r} | {atual!r}) = {pares[(atual, seguinte)]}/{total_atual} = {pares[(atual, seguinte)] / total_atual:.2f}")

# %% [markdown]
# ### Solução do Exercício 2

# %%
seguintes = {}
for atual, proxima in zip(palavras[:-1], palavras[1:]):
    seguintes.setdefault(atual, Counter())[proxima] += 1

gerador = np.random.default_rng(0)
palavra = "alice"
frase_gerada = [palavra]
for _ in range(40):
    opcoes = seguintes.get(palavra)
    if not opcoes:                 # palavra sem sucessor conhecido: recomeça
        palavra = "alice"
        continue
    candidatas = list(opcoes.keys())
    pesos = np.array(list(opcoes.values()), dtype=float)
    palavra = candidatas[gerador.choice(len(candidatas), p=pesos / pesos.sum())]
    frase_gerada.append(palavra)
print(" ".join(frase_gerada))

n_diferentes = len(contagem_palavras)
print(f"\n{n_diferentes} palavras diferentes: uma tabela de pares teria {n_diferentes**2:,} casas, "
      f"para só {len(palavras) - 1:,} pares no livro.")

# %% [markdown]
# O texto de palavras parece bem mais inglês: cada pedaço de duas palavras existe de verdade no livro. Mas o sentido se perde em
# poucas palavras, porque o modelo só lembra da última. E a tabela completa teria milhões de casas para uns 27 mil pares: quase
# todas vazias. Por isso o dicionário guarda só os pares que aparecem. (`setdefault` cria um `Counter` vazio na primeira vez que vê
# uma palavra.)

# %% [markdown]
# ### Solução do Exercício 3

# %%
contagens_tri = np.zeros((tamanho_vocabulario,) * 3)
np.add.at(contagens_tri, (indices_treino[:-2], indices_treino[1:-1], indices_treino[2:]), 1)
probabilidades_tri = (contagens_tri + 0.1) / (contagens_tri + 0.1).sum(axis=2, keepdims=True)

p_corretos_tri = probabilidades_tri[indices_validacao[:-2], indices_validacao[1:-1], indices_validacao[2:]]
print(f"perplexidade do trigrama: {np.exp(-np.mean(np.log(p_corretos_tri))):.1f}   "
      f"(bigrama: {perplexidade_bigrama(probabilidades, indices_validacao):.1f})")

gerador = np.random.default_rng(0)
contexto = list(codificar("th"))
for _ in range(300):
    p = probabilidades_tri[contexto[-2], contexto[-1]]
    contexto.append(gerador.choice(tamanho_vocabulario, p=p))
print(decodificar(contexto))

# %% [markdown]
# Com dois caracteres de memória, a perplexidade cai e o texto fica mais parecido com inglês: aparecem mais palavras reais.
# Mas a tabela já tem 104 mil casas para uns 130 mil trios de treino, a maioria vazia. A suavização precisa ser pequena (0,1); com 1,
# ela "espalharia" probabilidade demais pelas combinações nunca vistas. Com quatro ou cinco caracteres, a contagem deixaria de funcionar.
