# %% [markdown]
# # Módulo 09 — Linguagem 2: um mini modelo neural que escreve em inglês
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/09-modelo-neural-caracteres.ipynb)
#
# **Objetivo.** Trocar a tabela de contagens do Módulo 08 por uma rede neural que olha vários caracteres para trás.
# Você vai conhecer os embeddings, os vetores que representam cada caractere, e ver o que a rede aprendeu sobre eles.
#
# **O que você vai construir.** Um modelo de caracteres em PyTorch, no estilo do artigo de Bengio e colegas (2003), que lê
# *Alice no País das Maravilhas* e escreve um inglês bem mais convincente que o dos bigramas.

# %% [markdown]
# ## 1. A intuição: memória sem tabela gigante
#
# No Módulo 08, o bigrama olhava só o último caractere. Para olhar mais longe, a contagem não serve: com 5 caracteres de contexto,
# a tabela teria 229 milhões de casas, quase todas vazias.
#
# Em 2003, Yoshua Bengio e colegas propuseram outra saída (para palavras; aqui usamos caracteres):
#
# 1. cada token vira um **vetor** de poucos números, aprendido no treino: o embedding;
# 2. os vetores dos últimos $k$ tokens são colocados lado a lado e entram numa rede neural comum;
# 3. a rede devolve uma pontuação para cada token possível, e a softmax (Módulo 05) transforma isso em probabilidades.
#
# A grande vantagem: tokens que se comportam de forma parecida ganham vetores parecidos. O que a rede aprende sobre um contexto
# vale, em parte, para contextos semelhantes que ela nunca viu. A contagem não tem esse "jeitinho" de generalizar.

# %% [markdown]
# ## 2. O texto
#
# O mesmo livro e a mesma preparação do Módulo 08: tudo em minúsculas, um token por caractere, 90% para treino e 10% para validação.
# As funções abaixo foram copiadas do Módulo 08.

# %%
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

CAMINHO_LOCAL = "../dados/alice.txt"


def carregar_texto():
    """Vimos isso no Módulo 08: lê o livro de dados/ ou pega o mesmo texto no corpus Gutenberg da NLTK."""
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


texto = carregar_texto().lower()
vocabulario = sorted(set(texto))
tamanho_vocabulario = len(vocabulario)
indice_de = {caractere: i for i, caractere in enumerate(vocabulario)}
caractere_de = {i: caractere for i, caractere in enumerate(vocabulario)}


def codificar(trecho):
    """Vimos isso no Módulo 08: texto -> array de índices."""
    return np.array([indice_de[c] for c in trecho])


def decodificar(indices):
    """Vimos isso no Módulo 08: índices -> texto."""
    return "".join(caractere_de[int(i)] for i in indices)


indices_texto = codificar(texto)
corte = int(0.9 * len(indices_texto))
indices_treino, indices_validacao = indices_texto[:corte], indices_texto[corte:]
print(f"{len(texto)} caracteres, vocabulário de {tamanho_vocabulario}")

# %% [markdown]
# ## 3. A janela de contexto
#
# O modelo vai receber os últimos $k$ caracteres e prever o próximo. Esse número $k$ é o tamanho da janela de contexto
# (context window). Para treinar, deslizamos uma janela pelo texto e montamos pares (contexto, alvo).
#
# **Exemplo à mão**, com o texto "alice was" e $k = 3$:
#
# | contexto | alvo |
# |---|---|
# | "ali" | "c" |
# | "lic" | "e" |
# | "ice" | " " |
# | "ce " | "w" |
# | "e w" | "a" |
# | " wa" | "s" |

# %%
def montar_exemplos(indices, tamanho_contexto):
    """Desliza uma janela pelo texto: cada contexto de k índices tem como alvo o índice seguinte."""
    contextos, alvos = [], []
    for inicio in range(len(indices) - tamanho_contexto):
        contextos.append(indices[inicio:inicio + tamanho_contexto])
        alvos.append(indices[inicio + tamanho_contexto])
    return torch.tensor(np.array(contextos)), torch.tensor(np.array(alvos))


# Conferindo o exemplo feito à mão
contextos_mini, alvos_mini = montar_exemplos(codificar("alice was"), 3)
for contexto, alvo in zip(contextos_mini, alvos_mini):
    print(f"{decodificar(contexto)!r} -> {decodificar([alvo])!r}")

# %% [markdown]
# No modelo de verdade usamos $k = 8$: o modelo enxerga uma ou duas palavras para trás.

# %%
TAMANHO_CONTEXTO = 8
X_treino, Y_treino = montar_exemplos(indices_treino, TAMANHO_CONTEXTO)
X_validacao, Y_validacao = montar_exemplos(indices_validacao, TAMANHO_CONTEXTO)
print("treino:", tuple(X_treino.shape), "  validação:", tuple(X_validacao.shape))

# %% [markdown]
# ## 4. Embeddings: cada caractere vira um vetor
#
# Um embedding (embedding, algo como "incorporação") é uma tabela $C$ com uma linha por token. Cada linha é um vetor de $d$ números.
# Para representar um caractere, pegamos a sua linha. No começo os números são aleatórios; durante o treino, o gradiente descendente
# os ajusta como qualquer outro peso.
#
# **Por que isso é uma camada de rede?** Pegar a linha $i$ é o mesmo que multiplicar o vetor one-hot do token (Módulo 05) pela matriz $C$.
# Exemplo à mão, com 4 tokens e $d = 2$:
#
# $$ \begin{pmatrix} 0 & 0 & 1 & 0 \end{pmatrix} \begin{pmatrix} 0{,}1 & 0{,}5 \\ -0{,}3 & 0{,}2 \\ 0{,}8 & -0{,}4 \\ 0{,}0 & 0{,}9 \end{pmatrix} = \begin{pmatrix} 0{,}8 & -0{,}4 \end{pmatrix} $$
#
# O one-hot com 1 na posição 2 "escolhe" a linha 2 (contando do zero). Uma camada `nn.Linear` sem viés faria a mesma conta, só que
# gastando multiplicações com zeros. A camada `nn.Embedding` do PyTorch vai direto à linha certa.

# %%
C_exemplo = torch.tensor([[0.1, 0.5],
                          [-0.3, 0.2],
                          [0.8, -0.4],
                          [0.0, 0.9]])
one_hot = torch.tensor([0.0, 0.0, 1.0, 0.0])
print("one-hot @ C:     ", one_hot @ C_exemplo)
print("linha 2 de C:    ", C_exemplo[2])

embedding_exemplo = nn.Embedding(4, 2)
with torch.no_grad():
    embedding_exemplo.weight.copy_(C_exemplo)     # coloca a nossa tabela dentro da camada
print("nn.Embedding(2): ", embedding_exemplo(torch.tensor([2])))

# %% [markdown]
# ## 5. O modelo
#
# Para um contexto de $k = 8$ caracteres:
#
# 1. **embedding:** cada caractere vira um vetor de $d = 10$ números. Resultado: uma tabela $8 \times 10$;
# 2. **achatar:** os 8 vetores ficam lado a lado, formando um vetor de 80 números (como achatamos as imagens no Módulo 05);
# 3. **camada escondida:** 200 neurônios com ativação tanh;
# 4. **saída:** 47 pontuações (logits), uma por caractere do vocabulário.
#
# A tangente hiperbólica (hyperbolic tangent, tanh) é uma prima da sigmoide: tem o mesmo formato de S, mas vai de −1 a 1,
# com $\tanh(z) = 2\,\sigma(2z) - 1$. Foi a ativação usada por Bengio; a ReLU também funcionaria.
#
# Com as peças do Módulo 07, o modelo inteiro é um `nn.Sequential`. `nn.Flatten()` faz o passo 2.

# %%
DIM_EMBEDDING = 10
N_OCULTOS = 200


def criar_modelo(tamanho_contexto=TAMANHO_CONTEXTO, dim_embedding=DIM_EMBEDDING, n_ocultos=N_OCULTOS):
    """Modelo de caracteres no estilo de Bengio et al. (2003)."""
    return nn.Sequential(
        nn.Embedding(tamanho_vocabulario, dim_embedding),        # (lote, k)        -> (lote, k, d)
        nn.Flatten(),                                            # (lote, k, d)     -> (lote, k * d)
        nn.Linear(tamanho_contexto * dim_embedding, n_ocultos),  # (lote, k * d)    -> (lote, 200)
        nn.Tanh(),
        nn.Linear(n_ocultos, tamanho_vocabulario),               # (lote, 200)      -> (lote, 47)
    )


torch.manual_seed(0)
modelo = criar_modelo()
n_parametros = sum(p.numel() for p in modelo.parameters())
print(modelo)
print(f"{n_parametros} parâmetros (a tabela de trigramas do Módulo 08 tinha {tamanho_vocabulario**3})")

# %% [markdown]
# Com contexto de 8 caracteres e só uns 26 mil parâmetros. Uma tabela de contagens com o mesmo contexto teria $47^8$ casas,
# um número com 14 dígitos.

# %% [markdown]
# ## 6. Treinando
#
# A perda é a entropia cruzada (`nn.CrossEntropyLoss`, Módulo 07), e o otimizador é o Adam. A cada passo sorteamos um minilote
# de 128 exemplos. Nos últimos 25% dos passos, dividimos a taxa de aprendizado por 10: passos menores ajudam o modelo a se
# acomodar no fundo do vale, em vez de ficar pulando em volta dele.
#
# De tempos em tempos medimos a perda na validação. Para comparar com o Módulo 08, lembre: perplexidade $= e^{\text{perda}}$.
# O bigrama tinha perplexidade 11,1, ou seja, perda $\log 11{,}1 \approx 2{,}41$; o trigrama, 6,4 ($\approx 1{,}86$).

# %%
def treinar_modelo(modelo, passos=6000, tamanho_lote=128, taxa=0.003, semente=0, avaliar_a_cada=250):
    """Adam com minilotes sorteados; a taxa cai 10 vezes nos últimos 25% dos passos."""
    gerador = torch.Generator().manual_seed(semente)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    funcao_perda = nn.CrossEntropyLoss()
    historico = {"passo": [], "treino": [], "validacao": []}
    for passo in range(passos + 1):
        if passo == int(0.75 * passos):
            for grupo in otimizador.param_groups:
                grupo["lr"] = taxa / 10
        if passo % avaliar_a_cada == 0:
            with torch.no_grad():
                amostra = torch.randint(0, len(X_treino), (10000,), generator=gerador)
                historico["passo"].append(passo)
                historico["treino"].append(funcao_perda(modelo(X_treino[amostra]), Y_treino[amostra]).item())
                historico["validacao"].append(funcao_perda(modelo(X_validacao), Y_validacao).item())
        if passo == passos:
            break
        lote = torch.randint(0, len(X_treino), (tamanho_lote,), generator=gerador)
        perda = funcao_perda(modelo(X_treino[lote]), Y_treino[lote])
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    return historico


inicio = time.perf_counter()
historico = treinar_modelo(modelo)
print(f"treino em {time.perf_counter() - inicio:.0f} s")
perda_validacao = historico["validacao"][-1]
print(f"perda na validação: {perda_validacao:.3f}   perplexidade: {np.exp(perda_validacao):.2f}")

# %% [markdown]
# **O aprendizado acontecendo**, comparado com os modelos de contagem do Módulo 08 (linhas tracejadas).
# A perda de treino é medida numa amostra de 10 mil exemplos, para ser rápido.

# %%
plt.figure(figsize=(8, 4.5))
plt.plot(historico["passo"], historico["treino"], label="rede: treino")
plt.plot(historico["passo"], historico["validacao"], label="rede: validação")
plt.axhline(np.log(11.1), color="gray", linestyle="--", label="bigrama (Módulo 08)")
plt.axhline(np.log(6.4), color="black", linestyle="--", label="trigrama (Módulo 08)")
plt.xlabel("passo")
plt.ylabel("entropia cruzada")
plt.title("O modelo neural aprendendo a prever o próximo caractere")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Em poucas centenas de passos a rede passa do bigrama; depois, do trigrama. O degrau no fim é o efeito da taxa menor.
# A perda de validação fica um pouco acima da de treino: um leve sobreajuste (Módulo 05), esperado com um livro só.

# %% [markdown]
# ## 7. Escrevendo
#
# Para gerar texto, repetimos: pegar os últimos 8 caracteres, calcular as probabilidades, sortear o próximo (Módulo 08).
# A temperatura entra dividindo os logits antes da softmax: $\text{softmax}(z / T)$. É exatamente o $p^{1/T}$ normalizado do Módulo 08,
# escrito de outra forma. Se o começo tiver menos de 8 caracteres, completamos com espaços à esquerda (`rjust`).

# %%
def gerar_texto(modelo, inicio, n_caracteres, temperatura=1.0, semente=0):
    """Escreve n_caracteres, um de cada vez, a partir dos últimos k caracteres."""
    gerador = torch.Generator().manual_seed(semente)
    contexto = list(codificar(inicio.rjust(TAMANHO_CONTEXTO)))
    saida = inicio
    for _ in range(n_caracteres):
        with torch.no_grad():
            logits = modelo(torch.tensor([contexto[-TAMANHO_CONTEXTO:]]))[0]
        probabilidades = torch.softmax(logits / temperatura, dim=0)
        proximo = torch.multinomial(probabilidades, 1, generator=gerador).item()
        contexto.append(proximo)
        saida += caractere_de[proximo]
    return saida


for temperatura in [0.6, 1.0]:
    print(f"--- temperatura {temperatura} ---")
    print(gerar_texto(modelo, "alice was ", 400, temperatura=temperatura))
    print()

# %% [markdown]
# `torch.multinomial` sorteia uma posição de acordo com as probabilidades, como o `choice` do NumPy.
#
# Para comparar, o bigrama do Módulo 08, treinado no mesmo texto, com a mesma temperatura 1:

# %%
contagens = np.zeros((tamanho_vocabulario, tamanho_vocabulario))
np.add.at(contagens, (indices_treino[:-1], indices_treino[1:]), 1)
probabilidades_bigrama = (contagens + 1) / (contagens + 1).sum(axis=1, keepdims=True)

gerador_numpy = np.random.default_rng(0)
atual = indice_de[" "]
saida_bigrama = "alice was "
for _ in range(400):
    atual = gerador_numpy.choice(tamanho_vocabulario, p=probabilidades_bigrama[atual])
    saida_bigrama += caractere_de[atual]
print(saida_bigrama)

# %% [markdown]
# A diferença salta aos olhos. Com temperatura 0,6, o modelo neural escreve quase só palavras inglesas de verdade e usa as personagens
# do livro (*alice*, *queen*, *king*, *duchess*); com 1,0, inventa mais palavras, mas que "soam" inglesas. Ele também abre e fecha
# aspas e respeita a pontuação. O sentido ainda se perde em poucas palavras: oito
# caracteres de memória é pouco. Esse é o problema que o Módulo 10 ataca.

# %% [markdown]
# ## 8. O que a rede aprendeu sobre os caracteres
#
# Cada caractere tem agora um vetor de 10 números. Não dá para desenhar 10 dimensões, mas podemos projetá-las num plano, como a
# sombra de um objeto 3D numa parede. Escolhemos a "parede" que mostra os pontos mais espalhados: as duas direções de maior variação.
# Essa técnica é a análise de componentes principais (principal component analysis, PCA). O NumPy a calcula com `np.linalg.svd`;
# não precisa entender os detalhes agora, só a ideia da sombra.

# %%
tabela_embedding = modelo[0].weight.detach().numpy()           # forma (47, 10)
centralizada = tabela_embedding - tabela_embedding.mean(axis=0)
_, _, direcoes = np.linalg.svd(centralizada, full_matrices=False)
coordenadas = centralizada @ direcoes[:2].T                    # a "sombra" em 2D

vogais = set("aeiou")
fig, eixo = plt.subplots(figsize=(8, 7))
for i, caractere in caractere_de.items():
    if caractere in vogais:
        cor = "tab:red"
    elif caractere.isalpha():
        cor = "tab:blue"
    elif caractere.isdigit():
        cor = "tab:green"
    else:
        cor = "gray"
    rotulo = {" ": "␣", "\n": "\\n"}.get(caractere, caractere)
    eixo.text(coordenadas[i, 0], coordenadas[i, 1], rotulo, color=cor, fontsize=14, ha="center", va="center")
eixo.set_xlim(coordenadas[:, 0].min() - 0.5, coordenadas[:, 0].max() + 0.5)
eixo.set_ylim(coordenadas[:, 1].min() - 0.5, coordenadas[:, 1].max() + 0.5)
eixo.set_title("Embeddings em 2D: vogais (vermelho), consoantes (azul), algarismos (verde), outros (cinza)")
eixo.grid(True)
plt.show()

# %% [markdown]
# Ninguém disse à rede o que é pontuação ou vogal. Mesmo assim, os sinais de pontuação (? ! . , ; :) formam um grupo bem
# separado das letras, e as vogais mais comuns costumam ficar próximas umas das outras. A rede descobriu que caracteres que aparecem
# em contextos parecidos devem ter vetores parecidos, porque isso ajuda a prever o próximo. Já os algarismos, que aparecem pouquíssimas
# vezes no livro, ficam espalhados: a rede quase não os viu, e seus vetores mal saíram do sorteio inicial. Os grupos não são
# perfeitos e mudam um pouco a cada treino, mas a tendência aparece. Nos grandes modelos, os embeddings de **palavras** mostram o mesmo fenômeno em escala
# muito maior: "king" e "queen" ficam perto, "Paris" e "London" também.

# %% [markdown]
# ## 9. Comparando
#
# | modelo | contexto | parâmetros ou casas | perplexidade na validação |
# |---|---|---|---|
# | chute uniforme | 0 | 0 | 47 |
# | bigrama (Módulo 08) | 1 caractere | 2 209 | 11,1 |
# | trigrama (Módulo 08) | 2 caracteres | 103 823 | 6,4 |
# | rede neural (este módulo) | 8 caracteres | cerca de 26 mil | veja abaixo |

# %%
print(f"perplexidade da rede neural: {np.exp(perda_validacao):.2f}")

# %% [markdown]
# A rede olha mais longe com bem menos parâmetros do que a tabela de trigramas. O segredo está nos embeddings, que deixam a rede
# compartilhar o que aprendeu entre caracteres e contextos parecidos.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Escreva à mão os pares (contexto, alvo) do texto "the cat" com $k = 3$. Depois, com a matriz `C_exemplo`
# da seção 4, calcule à mão o produto do one-hot $(0, 1, 0, 0)$ por $C$. Confira as duas coisas com código.
#
# **Exercício 2 (médio).** Treine modelos com janelas de contexto $k$ = 1, 3 e 8, com 2000 passos cada (crie os exemplos de novo para
# cada $k$ e passe `tamanho_contexto` para `criar_modelo`). Compare as perplexidades de validação. O de $k = 1$ fica perto de qual
# modelo do Módulo 08? Por quê? (Dica: a função `treinar_modelo` usa as variáveis `X_treino` e `X_validacao`; troque-as antes de chamar.)
#
# **Exercício 3 (desafio).** Use o modelo treinado como "juiz de inglês". Escreva uma função que calcula a perda média (entropia cruzada)
# que o modelo dá a uma frase, caractere por caractere. Compare três frases: uma frase natural ("alice was very tired of sitting by her
# sister"), as mesmas palavras embaralhadas e as mesmas letras embaralhadas. Depois, mostre os 5 caracteres mais prováveis depois de
# "the queen sa" e de "said the ".

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | embedding | embedding (vetor de representação) | *Each character has a learned embedding.* |
# | context window | janela de contexto | *The context window has eight characters.* |
# | character-level model | modelo de caracteres | *A character-level model writes one letter at a time.* |
# | lookup table | tabela de consulta | *An embedding is a lookup table of vectors.* |
# | flatten | achatar | *Flatten the eight vectors into one long vector.* |
# | hyperbolic tangent | tangente hiperbólica | *The hyperbolic tangent goes from minus one to one.* |
# | logits | logits (pontuações) | *The model outputs 47 logits.* |
# | prompt | começo (texto inicial) | *The prompt was "alice was".* |
# | sample | sortear / amostra | *We sample the next character from the probabilities.* |
# | learning rate decay | redução da taxa de aprendizado | *Learning rate decay helps the model settle down.* |
# | principal component analysis | análise de componentes principais | *Principal component analysis projects the vectors onto a plane.* |
# | generalize | generalizar | *Embeddings help the model generalize to new contexts.* |
# | dimension | dimensão | *Each embedding has ten dimensions.* |

# %% [markdown]
# ## Resumo
#
# - Contar não escala para contextos longos; uma rede neural troca a tabela gigante por poucos parâmetros que generalizam.
# - A janela de contexto define quantos tokens anteriores o modelo enxerga; os exemplos de treino saem de uma janela deslizante.
# - Um embedding é uma tabela de vetores aprendidos, um por token; equivale a multiplicar o one-hot por uma matriz.
# - O modelo de Bengio (embeddings lado a lado, camada tanh, softmax) supera bigramas e trigramas e escreve palavras reais.
# - Projetados em 2D, os embeddings mostram grupos (como as vogais) que a rede descobriu sozinha.
#
# **Próximo módulo:** atenção e um Transformer em miniatura, a arquitetura por trás dos grandes modelos de linguagem.

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
# Pares: "the" → " ", "he " → "c", "e c" → "a", " ca" → "t". O one-hot $(0, 1, 0, 0)$ escolhe a linha 1 de $C$: $(-0{,}3;\ 0{,}2)$.

# %%
for contexto, alvo in zip(*montar_exemplos(codificar("the cat"), 3)):
    print(f"{decodificar(contexto)!r} -> {decodificar([alvo])!r}")
print(torch.tensor([0.0, 1.0, 0.0, 0.0]) @ C_exemplo)

# %% [markdown]
# ### Solução do Exercício 2

# %%
X_treino_original, Y_treino_original = X_treino, Y_treino
X_validacao_original, Y_validacao_original = X_validacao, Y_validacao

for k in [1, 3, 8]:
    X_treino, Y_treino = montar_exemplos(indices_treino, k)
    X_validacao, Y_validacao = montar_exemplos(indices_validacao, k)
    torch.manual_seed(0)
    modelo_k = criar_modelo(tamanho_contexto=k)
    historico_k = treinar_modelo(modelo_k, passos=2000)
    print(f"k = {k}: perplexidade de validação {np.exp(historico_k['validacao'][-1]):.2f}")

# devolve os dados originais (k = 8) para as próximas células
X_treino, Y_treino = X_treino_original, Y_treino_original
X_validacao, Y_validacao = X_validacao_original, Y_validacao_original

# %% [markdown]
# Com $k = 1$, a rede vê só o caractere atual: ela não pode fazer melhor que o bigrama, que já é a melhor tabela possível para
# um caractere de contexto. Por isso a perplexidade fica perto de 11. Com $k = 3$ a melhora é enorme. Com $k = 8$, em só 2000 passos,
# o modelo ainda não aproveitou o contexto maior e fica parecido com $k = 3$; com os 6000 passos da seção 6, chegou a cerca de 5,3.
# Mais contexto dá mais informação, mas exige mais treino (e, em geral, mais dados) para ser bem usado.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def perda_da_frase(modelo, frase):
    """Entropia cruzada média que o modelo dá a cada caractere da frase (o começo é completado com espaços)."""
    indices = codificar(" " * TAMANHO_CONTEXTO + frase)
    contextos, alvos = montar_exemplos(indices, TAMANHO_CONTEXTO)
    with torch.no_grad():
        return nn.CrossEntropyLoss()(modelo(contextos), alvos).item()


gerador_frases = np.random.default_rng(0)
natural = "alice was very tired of sitting by her sister"
palavras_embaralhadas = " ".join(gerador_frases.permutation(natural.split()))
letras_embaralhadas = "".join(gerador_frases.permutation(list(natural)))
for nome, frase in [("natural", natural), ("palavras embaralhadas", palavras_embaralhadas),
                    ("letras embaralhadas", letras_embaralhadas)]:
    print(f"{nome:22s} {perda_da_frase(modelo, frase):.2f}   {frase!r}")


def mais_provaveis(modelo, inicio, quantos=5):
    """Os caracteres mais prováveis depois de um começo, com as probabilidades."""
    contexto = torch.tensor(np.array([codificar(inicio.rjust(TAMANHO_CONTEXTO))[-TAMANHO_CONTEXTO:]]))
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(contexto)[0], dim=0)
    valores, posicoes = probabilidades.topk(quantos)
    return [(caractere_de[int(i)], round(v.item(), 3)) for v, i in zip(valores, posicoes)]


print(mais_provaveis(modelo, "the queen sa"))
print(mais_provaveis(modelo, "said the "))

# %% [markdown]
# A frase natural tem a menor perda: o modelo "acha" que ela é inglês. Com as palavras embaralhadas, cada palavra continua
# familiar, mas as junções entre elas são estranhas, e a perda sobe um pouco. Com as letras embaralhadas, a perda dispara.
# É o mesmo princípio que os grandes modelos usam para julgar a "naturalidade" de um texto. (`topk` devolve os maiores valores
# e as suas posições.)
