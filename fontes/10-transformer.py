# %% [markdown]
# # Módulo 10 — Linguagem 3: atenção e um Transformer em miniatura
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/10-transformer.ipynb)
#
# **Objetivo.** Entender a autoatenção (self-attention), a peça central dos grandes modelos de linguagem, calculando-a à mão,
# e montar um Transformer pequeno que escreve em inglês. No fim, ver como modelos como o ChatGPT e o Claude são criados.
#
# **O que você vai construir.** Um Transformer de caracteres com cerca de 160 mil parâmetros, treinado em CPU por uns 2 minutos
# com *Alice no País das Maravilhas*, e mapas que mostram para onde cada caractere "presta atenção".

# %% [markdown]
# ## 1. A intuição: o contexto importa
#
# Leia: *"Alice picked up the little bottle. On its label were the words DRINK ME, so she..."*. Para continuar, você precisa lembrar
# de coisas que ficaram para trás: quem é "she", o que é "its", o que estava escrito. A informação útil pode estar perto ou longe,
# e muda a cada palavra.
#
# O modelo do Módulo 09 tinha duas limitações:
#
# - enxergava só 8 caracteres para trás;
# - cada posição da janela tinha **pesos próprios** na camada escondida. Para enxergar 100 caracteres, a camada precisaria de
#   pesos para 100 posições, e o que ela aprende sobre "the" na posição 3 não vale para "the" na posição 50.
#
# A atenção (attention) resolve isso de outro jeito. Cada posição **olha para todas as anteriores** e decide, pelo conteúdo,
# quais importam agora. Os mesmos pesos servem para qualquer distância. É a ideia do artigo *Attention Is All You Need*
# (Vaswani e colegas, 2017), que criou o Transformer.

# %% [markdown]
# ## 2. A atenção como uma busca: consulta, chave e valor
#
# Pense numa biblioteca. Você chega com uma **consulta** (query): "livros sobre gatos". Cada livro tem uma **chave** (key), a etiqueta
# na lombada. Você compara a consulta com cada etiqueta e pega o **valor** (value), o conteúdo, dos livros que combinam mais.
#
# Na autoatenção, cada token faz os três papéis ao mesmo tempo. A partir do seu vetor, ele calcula:
#
# - $q$ (consulta): "o que eu estou procurando?";
# - $k$ (chave): "o que eu tenho a oferecer?";
# - $v$ (valor): "o que eu passo adiante se alguém me escolher".
#
# Para o token na posição $i$:
#
# 1. **combinação:** a pontuação com cada token $j$ é o produto escalar (Módulo 02) $q_i \cdot k_j$, dividido por $\sqrt{d}$,
#    onde $d$ é o tamanho dos vetores. Vetores que "apontam para o mesmo lado" dão pontuação alta;
# 2. **pesos:** a softmax (Módulo 05) transforma as pontuações em pesos positivos que somam 1;
# 3. **resultado:** a média dos valores, ponderada pelos pesos: $\sum_j \text{peso}_{ij}\, v_j$.
#
# **A máscara causal.** Um modelo que prevê o próximo caractere não pode espiar o futuro. Então o token $i$ só olha para
# $j \le i$. Na prática, colocamos $-\infty$ nas pontuações proibidas antes da softmax, pois $e^{-\infty} = 0$.
#
# **Por que dividir por $\sqrt{d}$?** O produto escalar soma $d$ termos; com vetores grandes, as pontuações crescem e a softmax fica
# "travada" num único token (o efeito da temperatura baixa do Módulo 08). Dividir por $\sqrt{d}$ mantém a escala sob controle.
#
# Em forma de matriz, para todos os tokens de uma vez (com $M$ sendo a máscara de zeros e $-\infty$):
#
# $$ \text{Atenção}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^T}{\sqrt{d}} + M \right) V $$

# %% [markdown]
# ## 3. Um exemplo minúsculo, à mão
#
# Três tokens, vetores de tamanho $d = 2$, com $q$, $k$ e $v$ já dados:
#
# | token | $q$ | $k$ | $v$ |
# |---|---|---|---|
# | 1 | (1, 0) | (1, 0) | (1, 0) |
# | 2 | (0, 1) | (1, 1) | (0, 1) |
# | 3 | (1, 1) | (0, 1) | (1, 1) |
#
# Usamos $\sqrt{2} \approx 1{,}414$.
#
# **Token 1.** Só pode olhar para si mesmo: peso 1, resultado $v_1 = (1, 0)$.
#
# **Token 2.** Olha para os tokens 1 e 2. Pontuações: $q_2 \cdot k_1 = 0$ e $q_2 \cdot k_2 = 1$; divididas por $\sqrt{2}$: $0$ e $0{,}707$.
# Softmax: $e^0 = 1$ e $e^{0{,}707} \approx 2{,}028$, soma $3{,}028$, pesos $\approx (0{,}330;\ 0{,}670)$.
# Resultado: $0{,}330\,(1, 0) + 0{,}670\,(0, 1) = (0{,}330;\ 0{,}670)$.
#
# **Token 3.** Olha para os três. Pontuações: $q_3 \cdot k_1 = 1$, $\;q_3 \cdot k_2 = 2$, $\;q_3 \cdot k_3 = 1$; divididas: $0{,}707$, $1{,}414$, $0{,}707$.
# Exponenciais: $2{,}028$, $4{,}113$, $2{,}028$, soma $8{,}169$, pesos $\approx (0{,}248;\ 0{,}503;\ 0{,}248)$.
# Resultado: $0{,}248\,(1, 0) + 0{,}503\,(0, 1) + 0{,}248\,(1, 1) \approx (0{,}497;\ 0{,}752)$.
#
# O token 3 prestou mais atenção ao token 2, cuja chave combinava melhor com a sua consulta. Agora, a mesma conta em PyTorch,
# com a fórmula de matriz.

# %%
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

pontuacoes = Q @ K.T / 2 ** 0.5
mascara = torch.tril(torch.ones(3, 3))                       # 1 onde pode olhar, 0 no futuro
pontuacoes = pontuacoes.masked_fill(mascara == 0, float("-inf"))
pesos = torch.softmax(pontuacoes, dim=1)
print("pesos de atenção (linha = quem olha, coluna = para quem):")
print(pesos.numpy().round(3))
print("resultados:")
print((pesos @ V).numpy().round(3))

# %% [markdown]
# `torch.tril` ("triangular inferior", lower triangle) cria a máscara: zeros acima da diagonal, onde ficaria o futuro.
# `masked_fill` troca as pontuações proibidas por $-\infty$.
#
# ## 4. De onde vêm $q$, $k$ e $v$; várias cabeças
#
# No exemplo, demos $q$, $k$ e $v$ prontos. No modelo, eles são calculados a partir do vetor $x$ de cada token por três matrizes de
# pesos aprendidas: $q = x W_Q$, $\; k = x W_K$, $\; v = x W_V$. O treino ajusta essas matrizes para que as consultas encontrem as
# chaves úteis.
#
# Uma única atenção procura uma coisa de cada vez. Por isso usamos várias em paralelo, as cabeças de atenção (attention heads),
# cada uma com as suas matrizes. Uma pode aprender a olhar o caractere anterior, outra o começo da palavra, outra as aspas abertas.
# Os resultados das cabeças são colocados lado a lado e misturados por uma camada linear.

# %% [markdown]
# ## 5. O bloco Transformer
#
# Um Transformer empilha vários blocos iguais. Cada bloco tem duas etapas:
#
# 1. **atenção** (várias cabeças): os tokens trocam informação entre si;
# 2. **rede densa** (feed-forward): cada token, sozinho, processa o que recebeu, numa rede com uma camada escondida (Módulo 04).
#
# E dois detalhes técnicos que fazem a pilha treinar bem:
#
# - **conexões residuais (residual connections):** em vez de $x \leftarrow f(x)$, fazemos $x \leftarrow x + f(x)$. Cada etapa só
#   **acrescenta** um ajuste. Na retropropagação, o gradiente tem um "atalho" direto pela soma, e não encolhe camada após camada
#   (lembre do gradiente que encolhia na sigmoide, no Módulo 04);
# - **normalização de camada (layer normalization):** antes de cada etapa, o vetor de cada token é padronizado para média 0 e desvio 1,
#   como os atributos do Módulo 02, com uma escala ajustável. Mantém os números em faixas boas.
#
# **E a ordem das palavras?** A atenção, sozinha, não sabe a ordem: é uma média ponderada, e "cão morde homem" e "homem morde cão"
# dariam as mesmas misturas. Por isso somamos ao embedding de cada token um **embedding de posição** (positional embedding), um vetor
# aprendido para a posição 0, outro para a 1, e assim por diante.
#
# O modelo completo:
#
# 1. embedding do token + embedding da posição;
# 2. alguns blocos (atenção + rede densa, com residuais e normalização);
# 3. normalização final e uma camada linear que dá as 47 pontuações do próximo caractere.
#
# Uma vantagem enorme: o modelo prevê o próximo caractere **em todas as posições ao mesmo tempo**. Uma sequência de 64 caracteres
# vira 64 exemplos de treino de uma vez (a máscara garante que cada posição só usa o passado).

# %% [markdown]
# ## 6. O texto
#
# O mesmo livro e a mesma preparação dos Módulos 08 e 09 (funções copiadas do Módulo 08).

# %%
CAMINHO_LOCAL = "../dados/alice.txt"
URL_DO_TEXTO = "https://raw.githubusercontent.com/LucianoBrazil/curso-ia/main/dados/alice.txt"


def carregar_texto():
    """Vimos isso no Módulo 08: lê o livro de dados/, baixa do GitHub ou usa a cópia da NLTK."""
    if os.path.exists(CAMINHO_LOCAL):
        with open(CAMINHO_LOCAL, encoding="utf-8") as arquivo:
            return arquivo.read()
    try:
        with urllib.request.urlopen(URL_DO_TEXTO, timeout=30) as resposta:
            return resposta.read().decode("utf-8")
    except OSError:                                       # sem acesso ao GitHub: plano B, a cópia da NLTK
        pass
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
    """Vimos isso no Módulo 08: texto -> lista de índices."""
    return [indice_de[c] for c in trecho]


def decodificar(indices):
    """Vimos isso no Módulo 08: índices -> texto."""
    return "".join(caractere_de[int(i)] for i in indices)


dados = torch.tensor(codificar(texto))
corte = int(0.9 * len(dados))
dados_treino, dados_validacao = dados[:corte], dados[corte:]
print(f"treino: {len(dados_treino)} caracteres   validação: {len(dados_validacao)}   vocabulário: {tamanho_vocabulario}")

# %% [markdown]
# Para treinar, sorteamos trechos de 64 caracteres. A entrada é o trecho; o alvo é o mesmo trecho deslocado uma posição
# (em cada posição, o caractere seguinte). `torch.stack` empilha os trechos numa tabela.

# %%
TAMANHO_CONTEXTO = 64


def sortear_lote(dados, tamanho_lote, gerador):
    """Trechos aleatórios de TAMANHO_CONTEXTO caracteres e, como alvo, os mesmos trechos uma posição à frente."""
    inicios = torch.randint(0, len(dados) - TAMANHO_CONTEXTO - 1, (tamanho_lote,), generator=gerador)
    entradas = torch.stack([dados[i:i + TAMANHO_CONTEXTO] for i in inicios])
    alvos = torch.stack([dados[i + 1:i + TAMANHO_CONTEXTO + 1] for i in inicios])
    return entradas, alvos


exemplo_entrada, exemplo_alvo = sortear_lote(dados_treino, 1, torch.Generator().manual_seed(0))
print("entrada:", repr(decodificar(exemplo_entrada[0][:30])))
print("alvo:   ", repr(decodificar(exemplo_alvo[0][:30])))

# %% [markdown]
# ## 7. O Transformer em código
#
# Até aqui montamos redes com `nn.Sequential`. A atenção precisa de um caminho que não é uma simples sequência (as somas residuais,
# a máscara), então usamos uma **classe** (class). Uma classe é um molde que junta as peças e o jeito de usá-las:
#
# - `__init__` cria as peças (as camadas), uma vez só;
# - `forward` diz como os dados passam pelas peças. Quando chamamos `modelo(x)`, o PyTorch executa o `forward`;
# - `self` é o próprio objeto: `self.consulta` é "a camada consulta deste objeto";
# - `class CabecaAtencao(nn.Module)` quer dizer que a nossa classe é um tipo de `nn.Module`, e herda dele tudo o que o PyTorch precisa
#   (guardar os parâmetros, calcular gradientes). Por isso a primeira linha do `__init__` é `super().__init__()`.
#
# Os parâmetros `usar_mascara` e `usar_posicao` existem só para os exercícios.

# %%
class CabecaAtencao(nn.Module):
    """Uma cabeça de autoatenção causal: a fórmula softmax(QKᵀ/√d + M) V da seção 2."""

    def __init__(self, dim_modelo, dim_cabeca, usar_mascara=True):
        super().__init__()
        self.consulta = nn.Linear(dim_modelo, dim_cabeca, bias=False)   # W_Q
        self.chave = nn.Linear(dim_modelo, dim_cabeca, bias=False)      # W_K
        self.valor = nn.Linear(dim_modelo, dim_cabeca, bias=False)      # W_V
        self.usar_mascara = usar_mascara
        self.ultimos_pesos = None                                        # guardados para desenhar depois

    def forward(self, x):
        n_posicoes = x.shape[1]
        Q, K, V = self.consulta(x), self.chave(x), self.valor(x)
        pontuacoes = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5
        if self.usar_mascara:
            mascara = torch.tril(torch.ones(n_posicoes, n_posicoes))
            pontuacoes = pontuacoes.masked_fill(mascara == 0, float("-inf"))
        pesos = torch.softmax(pontuacoes, dim=-1)
        self.ultimos_pesos = pesos.detach()
        return pesos @ V


class AtencaoVariasCabecas(nn.Module):
    """Várias cabeças em paralelo; os resultados ficam lado a lado e passam por uma camada linear."""

    def __init__(self, dim_modelo, n_cabecas, usar_mascara=True):
        super().__init__()
        self.cabecas = nn.ModuleList([CabecaAtencao(dim_modelo, dim_modelo // n_cabecas, usar_mascara)
                                      for _ in range(n_cabecas)])
        self.mistura = nn.Linear(dim_modelo, dim_modelo)

    def forward(self, x):
        return self.mistura(torch.cat([cabeca(x) for cabeca in self.cabecas], dim=-1))


class BlocoTransformer(nn.Module):
    """Atenção e rede densa, cada uma com normalização antes e conexão residual em volta."""

    def __init__(self, dim_modelo, n_cabecas, usar_mascara=True):
        super().__init__()
        self.normalizacao_1 = nn.LayerNorm(dim_modelo)
        self.atencao = AtencaoVariasCabecas(dim_modelo, n_cabecas, usar_mascara)
        self.normalizacao_2 = nn.LayerNorm(dim_modelo)
        self.rede_densa = nn.Sequential(
            nn.Linear(dim_modelo, 4 * dim_modelo), nn.ReLU(), nn.Linear(4 * dim_modelo, dim_modelo))

    def forward(self, x):
        x = x + self.atencao(self.normalizacao_1(x))     # conexão residual
        x = x + self.rede_densa(self.normalizacao_2(x))  # conexão residual
        return x


class MiniTransformer(nn.Module):
    """Embeddings de token e de posição, blocos Transformer e uma camada de saída com uma pontuação por caractere."""

    def __init__(self, dim_modelo=64, n_cabecas=4, n_blocos=3, usar_mascara=True, usar_posicao=True):
        super().__init__()
        self.embedding_token = nn.Embedding(tamanho_vocabulario, dim_modelo)
        self.embedding_posicao = nn.Embedding(TAMANHO_CONTEXTO, dim_modelo)
        self.blocos = nn.Sequential(*[BlocoTransformer(dim_modelo, n_cabecas, usar_mascara) for _ in range(n_blocos)])
        self.normalizacao_final = nn.LayerNorm(dim_modelo)
        self.saida = nn.Linear(dim_modelo, tamanho_vocabulario)
        self.usar_posicao = usar_posicao

    def forward(self, indices):
        x = self.embedding_token(indices)                                    # (lote, posições, dim)
        if self.usar_posicao:
            x = x + self.embedding_posicao(torch.arange(indices.shape[1]))
        return self.saida(self.normalizacao_final(self.blocos(x)))           # (lote, posições, 47)


torch.manual_seed(0)
modelo = MiniTransformer()
print(f"{sum(p.numel() for p in modelo.parameters())} parâmetros")

# %% [markdown]
# (`nn.ModuleList` é uma lista de camadas que o PyTorch reconhece como parte do modelo. O `*` em `nn.Sequential(*[...])` "desempacota"
# a lista, passando cada bloco como um argumento separado.)

# %% [markdown]
# ## 8. Treinando
#
# O laço é o do Módulo 09, com duas novidades:
#
# - a perda é a entropia cruzada em **todas** as posições. A saída tem forma (lote, 64, 47); `.reshape(-1, 47)` a transforma numa
#   tabela com uma linha por posição, como a `nn.CrossEntropyLoss` espera;
# - o otimizador é o **AdamW**: o Adam com um pequeno "puxão" de todos os pesos em direção a zero a cada passo. Esse decaimento de
#   pesos (weight decay) é uma forma de regularização, a técnica contra o sobreajuste prometida no Módulo 05: pesos menores tendem a
#   dar modelos mais "suaves", que decoram menos.
#
# São 2000 passos com lotes de 32 trechos. No Colab gratuito, isso leva uns 2 a 3 minutos.

# %%
funcao_perda = nn.CrossEntropyLoss()
gerador_avaliacao = torch.Generator().manual_seed(1)
lote_avaliacao_treino = sortear_lote(dados_treino, 200, gerador_avaliacao)
lote_avaliacao_validacao = sortear_lote(dados_validacao, 200, gerador_avaliacao)


def perda_media(modelo, lote):
    """Entropia cruzada em todas as posições de um lote fixo."""
    entradas, alvos = lote
    with torch.no_grad():
        return funcao_perda(modelo(entradas).reshape(-1, tamanho_vocabulario), alvos.reshape(-1)).item()


def treinar_transformer(modelo, passos=2000, tamanho_lote=32, taxa=0.003, fracao_reducao=0.8, avaliar_a_cada=250, semente=0):
    """AdamW com lotes sorteados; a taxa cai 10 vezes a partir de fracao_reducao dos passos."""
    gerador = torch.Generator().manual_seed(semente)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=0.01)
    historico = {"passo": [], "treino": [], "validacao": []}
    for passo in range(passos + 1):
        if passo == int(fracao_reducao * passos):
            for grupo in otimizador.param_groups:
                grupo["lr"] = taxa / 10
        if passo % avaliar_a_cada == 0:
            historico["passo"].append(passo)
            historico["treino"].append(perda_media(modelo, lote_avaliacao_treino))
            historico["validacao"].append(perda_media(modelo, lote_avaliacao_validacao))
        if passo == passos:
            break
        entradas, alvos = sortear_lote(dados_treino, tamanho_lote, gerador)
        perda = funcao_perda(modelo(entradas).reshape(-1, tamanho_vocabulario), alvos.reshape(-1))
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    return historico


inicio = time.perf_counter()
historico = treinar_transformer(modelo)
print(f"treino em {time.perf_counter() - inicio:.0f} s")
perda_validacao = historico["validacao"][-1]
print(f"perda na validação: {perda_validacao:.3f}   perplexidade: {np.exp(perda_validacao):.2f}")

# %% [markdown]
# **O aprendizado acontecendo**, com os modelos anteriores como referência (perda $= \log$ da perplexidade).

# %%
plt.figure(figsize=(8, 4.5))
plt.plot(historico["passo"], historico["treino"], marker="o", label="Transformer: treino")
plt.plot(historico["passo"], historico["validacao"], marker="o", label="Transformer: validação")
plt.axhline(np.log(11.1), color="lightgray", linestyle="--", label="bigrama (Módulo 08)")
plt.axhline(np.log(6.4), color="gray", linestyle="--", label="trigrama (Módulo 08)")
plt.axhline(np.log(5.29), color="black", linestyle="--", label="rede de Bengio (Módulo 09)")
plt.ylim(1.0, 3.0)
plt.xlabel("passo")
plt.ylabel("entropia cruzada")
plt.title("O Transformer aprendendo")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# O Transformer passa a rede do Módulo 09. A distância entre treino e validação mostra um começo de sobreajuste: com um livro só,
# um modelo maior logo começa a decorar. Os grandes modelos treinam com tanto texto que quase nunca veem o mesmo trecho duas vezes.

# %% [markdown]
# ## 9. Escrevendo
#
# A geração é como no Módulo 09, só que agora o contexto pode ter até 64 caracteres: a cada passo, entregamos os últimos 64 ao modelo
# e usamos a pontuação da **última** posição. Gerar um token de cada vez, sempre realimentando o que já foi escrito, chama-se geração
# autorregressiva (autoregressive generation). É exatamente assim que os assistentes de IA escrevem as respostas.

# %%
def gerar_texto(modelo, inicio, n_caracteres, temperatura=0.8, semente=0):
    """Gera um caractere de cada vez, olhando os últimos TAMANHO_CONTEXTO caracteres."""
    gerador = torch.Generator().manual_seed(semente)
    contexto = codificar(inicio)
    for _ in range(n_caracteres):
        entrada = torch.tensor([contexto[-TAMANHO_CONTEXTO:]])
        with torch.no_grad():
            logits = modelo(entrada)[0, -1]
        probabilidades = torch.softmax(logits / temperatura, dim=0)
        contexto.append(torch.multinomial(probabilidades, 1, generator=gerador).item())
    return decodificar(contexto)


print(gerar_texto(modelo, "alice was ", 600))

# %% [markdown]
# Compare com o texto do Módulo 09. Mais palavras certas, frases com mais "cara" de frase, diálogos entre aspas, expressões típicas do
# livro (*said alice to herself*, *the queen*). Ainda sem sentido de verdade: o modelo é minúsculo e leu um único livro.

# %% [markdown]
# ## 10. Para onde o modelo olha
#
# Cada cabeça guardou seus pesos de atenção da última passagem. Vamos passar uma frase pelo modelo e desenhar os pesos das quatro cabeças
# do **primeiro** bloco. Cada linha é um caractere "olhando"; cada coluna, um caractere "olhado". O triângulo vazio é a máscara causal.

# %%
frase = "the queen said to alice"
with torch.no_grad():
    modelo(torch.tensor([codificar(frase)]))

rotulos = ["␣" if c == " " else c for c in frase]
fig, eixos = plt.subplots(1, 4, figsize=(20, 5.5))
for numero, (eixo, cabeca) in enumerate(zip(eixos, modelo.blocos[0].atencao.cabecas)):
    eixo.imshow(cabeca.ultimos_pesos[0].numpy(), cmap="Blues")
    eixo.set_xticks(range(len(frase)), rotulos, fontsize=8)
    eixo.set_yticks(range(len(frase)), rotulos, fontsize=8)
    eixo.set_title(f"bloco 1, cabeça {numero + 1}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Cada cabeça desenvolveu um "estilo" próprio. Na execução feita ao preparar este módulo (a sua pode variar um pouco), a cabeça 1 olha
# quase só para o caractere imediatamente anterior: é a diagonal logo abaixo da principal. A cabeça 2 olha para os últimos poucos
# caracteres, com destaque para alguns espaços, os limites das palavras. As cabeças 3 e 4 espalham a atenção por toda a frase, como uma
# média do que veio antes. Ninguém programou isso: os padrões surgiram porque ajudam a prever o próximo caractere.
# Interpretar o que cada cabeça faz é um campo de pesquisa por si só, chamado interpretabilidade (interpretability).

# %% [markdown]
# ## 11. A escada dos nossos modelos
#
# | modelo | módulo | o que olha | perplexidade na validação |
# |---|---|---|---|
# | chute uniforme | 08 | nada | 47 |
# | bigrama | 08 | 1 caractere | 11,1 |
# | trigrama | 08 | 2 caracteres | 6,4 |
# | rede de Bengio | 09 | 8 caracteres, posições fixas | 5,3 |
# | Transformer | 10 | até 64 caracteres, com atenção | veja abaixo |

# %%
print(f"perplexidade do Transformer: {np.exp(perda_validacao):.2f}")

# %% [markdown]
# ## 12. Como modelos como o ChatGPT e o Claude são criados
#
# O seu Transformer e os grandes modelos de linguagem (large language models, LLMs) são parentes próximos: mesma ideia, mesma
# arquitetura básica, mesma tarefa de prever o próximo token. A diferença está na escala e nas etapas depois do pré-treinamento.
#
# **1. Pré-treinamento (pretraining).** Exatamente o que você fez na seção 8, em outra escala. Em vez de um livro, trilhões de tokens
# de textos da internet, livros, artigos e código. Em vez de caracteres, pedaços de palavras (Módulo 08). Em vez de 160 mil parâmetros,
# dezenas ou centenas de bilhões (os números exatos de muitos modelos comerciais não são públicos). Em vez de 2 minutos numa CPU, meses
# em milhares de processadores gráficos (GPUs). O resultado é um modelo que sabe continuar textos de quase qualquer tipo, mas que ainda não
# é um assistente: se você escrever uma pergunta, ele pode continuar com outras perguntas, como num formulário.
#
# **2. Ajuste fino (fine-tuning) com instruções.** O modelo continua sendo treinado, agora com exemplos de conversas: uma pergunta ou
# pedido e uma boa resposta, muitos escritos por pessoas. Ele aprende o formato "usuário pergunta, assistente responde".
#
# **3. Aprendizado por reforço com feedback humano (reinforcement learning from human feedback, RLHF).** Pessoas comparam duas respostas do
# modelo e dizem qual é melhor (mais útil, mais correta, mais segura). Com essas comparações, treina-se um segundo modelo, o modelo de
# recompensa (reward model), que aprende a dar notas às respostas. Depois, o modelo de linguagem é ajustado para produzir respostas que
# recebam notas altas. Uma variação, usada pela Anthropic no Claude, é a IA Constitucional (Constitutional AI): parte das avaliações é feita
# pela própria IA, guiada por uma lista de princípios escritos.
#
# No fundo, cada etapa usa o que você aprendeu: uma função de perda, gradientes calculados pela retropropagação e um otimizador dando passos.
#
# ### Limites que todo usuário deve conhecer
#
# - **Alucinações (hallucinations).** O modelo é treinado para produzir texto **provável**, não texto **verdadeiro**. O seu Transformer inventa
#   palavras que "soam" inglesas; um grande modelo pode inventar fatos, datas, citações e referências com a mesma fluência com que diz coisas certas.
#   Para informações importantes, confira em fontes confiáveis.
# - **Conhecimento com data de corte.** O modelo sabe o que estava nos dados de treino; fatos posteriores ficam de fora, a menos que ele tenha
#   acesso a ferramentas como busca.
# - **Vieses.** O modelo aprende os padrões dos textos que leu, inclusive preconceitos e erros.
# - **Raciocínio frágil.** Ele pode errar contas e lógica simples, sobretudo em problemas diferentes dos que viu. Pedir o passo a passo ajuda,
#   mas não garante.
# - **Janela de contexto limitada.** Como o seu modelo, que via 64 caracteres, os grandes modelos também têm um limite (bem maior) de texto que
#   conseguem considerar de uma vez.
#
# ### Próximos passos de estudo
#
# - **Andrej Karpathy, *Neural Networks: Zero to Hero*** (vídeos gratuitos no YouTube, em inglês): constrói do zero, em código, de um modelo de
#   bigramas até um GPT. Combina perfeitamente com este curso e treina o seu inglês.
# - **3Blue1Brown, série *Neural Networks*** (YouTube, em inglês, com legendas): animações excelentes sobre redes, retropropagação e Transformers.
# - **O artigo *Attention Is All You Need*** (Vaswani et al., 2017): leitura técnica, mas agora você reconhece as peças.
# - **Livro *Dive into Deep Learning*** (gratuito em d2l.ai, com código): vai mais fundo em cada tema, com exercícios.
# - **Livro *Deep Learning*** (Goodfellow, Bengio e Courville, gratuito em deeplearningbook.org): a referência matemática.
# - **Este curso:** os módulos bônus (o pêndulo, redes convolucionais para imagens e um projeto seu) estão disponíveis se você quiser continuar.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** No exemplo da seção 3, troque a consulta do token 2 por $q_2 = (1, 0)$. Calcule à mão os pesos de atenção do
# token 2 e o seu resultado. Confira com o código da seção 3.
#
# **Exercício 2 (médio).** Treine um `MiniTransformer(usar_posicao=False)` por 500 passos, sem reduzir a taxa (`fracao_reducao=1.0`), e compare
# a perda de validação com a do modelo principal no passo 500 (está em `historico`). Gere um texto com ele. O que a falta da informação de
# posição atrapalha?
#
# **Exercício 3 (desafio).** Treine um `MiniTransformer(usar_mascara=False)` por 300 passos (`fracao_reducao=1.0`). Compare a perda de **treino**
# com a do modelo principal no passo 250. Depois gere texto com ele. Explique o que aconteceu. (Pense: sem a máscara, o que a posição $i$
# consegue ver durante o treino? E durante a geração?)

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | attention | atenção | *Attention lets each token look at the others.* |
# | self-attention | autoatenção | *In self-attention, the queries, keys and values come from the same text.* |
# | query, key, value | consulta, chave, valor | *The query is compared with every key.* |
# | attention head | cabeça de atenção | *Each attention head learns a different pattern.* |
# | causal mask | máscara causal | *The causal mask hides the future.* |
# | residual connection | conexão residual | *A residual connection adds the input back to the output.* |
# | layer normalization | normalização de camada | *Layer normalization keeps the numbers in a good range.* |
# | positional embedding | embedding de posição | *Positional embeddings tell the model the order of the tokens.* |
# | autoregressive | autorregressivo | *The model generates text in an autoregressive way.* |
# | weight decay | decaimento de pesos | *Weight decay pulls the weights towards zero.* |
# | pretraining | pré-treinamento | *Pretraining uses trillions of tokens.* |
# | fine-tuning | ajuste fino | *Fine-tuning teaches the model to follow instructions.* |
# | reward model | modelo de recompensa | *The reward model scores each answer.* |
# | hallucination | alucinação | *A hallucination is a confident but false answer.* |
# | large language model | grande modelo de linguagem | *A large language model has billions of parameters.* |

# %% [markdown]
# ## Resumo
#
# - Na autoatenção, cada token compara a sua consulta com as chaves dos anteriores e faz uma média ponderada dos valores: $\text{softmax}(QK^T/\sqrt{d} + M)V$.
# - A máscara causal impede o modelo de espiar o futuro; várias cabeças procuram coisas diferentes ao mesmo tempo.
# - Um bloco Transformer junta atenção e rede densa, com conexões residuais, normalização de camada e embeddings de posição.
# - O mini Transformer, treinado em 2 minutos, supera todos os modelos anteriores do curso e escreve um inglês ainda mais convincente.
# - Os grandes modelos usam a mesma receita em escala gigante, mais ajuste fino e RLHF; e podem alucinar, porque buscam o provável, não o verdadeiro.
#
# **Próximo passo:** este é o último módulo do plano principal. Parabéns por chegar até aqui! Os módulos bônus estão disponíveis se você quiser continuar.

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
# Com $q_2 = (1, 0)$: $q_2 \cdot k_1 = 1$ e $q_2 \cdot k_2 = 1$. Pontuações iguais dão pesos iguais: $(0{,}5;\ 0{,}5)$.
# Resultado: $0{,}5\,(1, 0) + 0{,}5\,(0, 1) = (0{,}5;\ 0{,}5)$.

# %%
Q_ex1 = Q.clone()
Q_ex1[1] = torch.tensor([1.0, 0.0])
pesos_ex1 = torch.softmax((Q_ex1 @ K.T / 2 ** 0.5).masked_fill(mascara == 0, float("-inf")), dim=1)
print("pesos do token 2:", pesos_ex1[1].numpy().round(3), "  resultado:", (pesos_ex1 @ V)[1].numpy().round(3))

# %% [markdown]
# ### Solução do Exercício 2

# %%
torch.manual_seed(0)
modelo_sem_posicao = MiniTransformer(usar_posicao=False)
historico_sem_posicao = treinar_transformer(modelo_sem_posicao, passos=500, fracao_reducao=1.0)
indice_500 = historico["passo"].index(500)
print(f"perda de validação no passo 500: com posição {historico['validacao'][indice_500]:.3f}   "
      f"sem posição {historico_sem_posicao['validacao'][-1]:.3f}")
print(gerar_texto(modelo_sem_posicao, "alice was ", 300))

# %% [markdown]
# Sem embeddings de posição, a perda fica mais alta. O modelo ainda tem alguma pista de ordem (a máscara causal faz cada posição ver um número
# diferente de caracteres), mas não sabe direito **onde** está cada caractere que ele olha: "ab" e "ba" ficam parecidos para ele. Para
# soletrar palavras, onde a ordem das letras é tudo, isso atrapalha bastante.

# %% [markdown]
# ### Solução do Exercício 3

# %%
torch.manual_seed(0)
modelo_sem_mascara = MiniTransformer(usar_mascara=False)
historico_sem_mascara = treinar_transformer(modelo_sem_mascara, passos=300, fracao_reducao=1.0, avaliar_a_cada=50)
indice_250 = historico["passo"].index(250)
print(f"perda de treino no passo 250: com máscara {historico['treino'][indice_250]:.3f}   "
      f"sem máscara {historico_sem_mascara['treino'][historico_sem_mascara['passo'].index(250)]:.3f}")
print(gerar_texto(modelo_sem_mascara, "alice was ", 300))

# %% [markdown]
# Sem a máscara, a perda de treino despenca: a posição $i$ consegue **ver** o caractere $i+1$, que é justamente a resposta. O modelo aprende
# a "colar", copiando o vizinho da frente. Na geração, porém, o próximo caractere ainda não existe; na última posição não há nada para copiar,
# e o texto sai sem sentido. Uma perda de treino ótima não garante um bom modelo: é preciso que o treino se pareça com o uso real.
# A máscara causal é o que garante isso.
