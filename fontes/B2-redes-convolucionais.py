# %% [markdown]
# # Módulo B2 (bônus) — Redes convolucionais para imagens
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B2-redes-convolucionais.ipynb)
#
# **Objetivo.** Entender a convolução, a operação por trás das redes que reconhecem imagens, calculando-a à mão, e ver por que ela é mais
# adequada a imagens do que as camadas densas dos Módulos 04 a 07.
#
# **O que você vai construir.** Filtros feitos à mão que detectam bordas, uma rede convolucional (CNN) em PyTorch para os dígitos do
# Módulo 05 e um teste que mostra a vantagem dela quando os dígitos estão fora do lugar.

# %% [markdown]
# ## 1. A intuição: procurar o mesmo padrão em todo lugar
#
# A rede do Módulo 05 "achatava" a imagem num vetor de 64 números. Dois problemas:
#
# - **a vizinhança se perde.** Para a camada densa, o pixel da linha 3, coluna 4, é só a entrada número 28. Ela não sabe que ele é vizinho
#   do pixel de cima ou do lado;
# - **o que se aprende num lugar não vale em outro.** Se a rede aprende a reconhecer um traço vertical no canto esquerdo, precisa aprender
#   de novo, com outros pesos, o mesmo traço no canto direito.
#
# A rede convolucional (convolutional neural network, CNN) muda a estratégia. Ela usa pequenos **filtros** (por exemplo, $3 \times 3$) que
# percorrem a imagem inteira, como uma lupa que passa por cada pedaço procurando um padrão: uma borda, uma curva, uma ponta. O **mesmo**
# filtro, com os **mesmos** pesos, é usado em todas as posições. Isso se chama compartilhamento de pesos (weight sharing).

# %% [markdown]
# ## 2. A convolução, à mão
#
# Colocamos o filtro $K$ sobre um pedaço da imagem $I$ do mesmo tamanho, multiplicamos posição por posição e somamos tudo (é um produto
# escalar, Módulo 02). Depois deslizamos o filtro uma casa e repetimos. Para um filtro $3 \times 3$:
#
# $$ S[i, j] = \sum_{m=0}^{2} \sum_{n=0}^{2} I[i + m,\ j + n] \cdot K[m, n] $$
#
# O resultado $S$ é um mapa de ativação (feature map): ele diz **onde** o padrão do filtro aparece. (Os matemáticos chamariam esta conta
# de correlação cruzada; a convolução "de livro" vira o filtro ao contrário. Em redes neurais, o nome convolução ficou para as duas, e como os
# filtros são aprendidos, tanto faz.)
#
# **Exemplo à mão.** Uma imagem $5 \times 5$, escura à esquerda (0) e clara à direita (1), e um filtro que compara a coluna da direita com a da
# esquerda:
#
# $$ I = \begin{pmatrix} 0&0&0&1&1 \\ 0&0&0&1&1 \\ 0&0&0&1&1 \\ 0&0&0&1&1 \\ 0&0&0&1&1 \end{pmatrix} \qquad K = \begin{pmatrix} -1&0&1 \\ -1&0&1 \\ -1&0&1 \end{pmatrix} $$
#
# - posição (0, 0), colunas 0 a 2: cada linha dá $-1 \cdot 0 + 0 \cdot 0 + 1 \cdot 0 = 0$; total $0$;
# - posição (0, 1), colunas 1 a 3: cada linha dá $-1 \cdot 0 + 0 \cdot 0 + 1 \cdot 1 = 1$; três linhas, total $3$;
# - posição (0, 2), colunas 2 a 4: cada linha dá $-1 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 = 1$; total $3$.
#
# O mapa tem zeros onde a imagem é uniforme e valores altos onde há uma borda vertical. Com um filtro $3 \times 3$ numa imagem $5 \times 5$,
# o mapa fica $3 \times 3$: o filtro cabe em 3 posições por linha.
#
# **Ligação com o Módulo 01.** Esse filtro calcula "valor à direita menos valor à esquerda": é o passinho para cada lado,
# $f(x + h) - f(x - h)$, a derivada aproximada da intensidade ao longo da horizontal. Detectar bordas é procurar onde a imagem muda depressa,
# ou seja, onde a derivada é grande.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import load_digits


def convoluir(imagem, filtro):
    """Convolução (correlação cruzada) sem borda extra: o filtro percorre todas as posições em que cabe."""
    altura_f, largura_f = filtro.shape
    altura_s = imagem.shape[0] - altura_f + 1
    largura_s = imagem.shape[1] - largura_f + 1
    saida = np.zeros((altura_s, largura_s))
    for i in range(altura_s):
        for j in range(largura_s):
            saida[i, j] = np.sum(imagem[i:i + altura_f, j:j + largura_f] * filtro)
    return saida


imagem_borda = np.array([[0, 0, 0, 1, 1]] * 5, dtype=float)
filtro_vertical = np.array([[-1, 0, 1]] * 3, dtype=float)
print(convoluir(imagem_borda, filtro_vertical))

# %% [markdown]
# ## 3. Filtros num dígito de verdade
#
# Vamos aplicar dois filtros feitos à mão a um dígito do `load_digits` (Módulo 05): o de bordas verticais e a sua versão "deitada", que detecta
# bordas horizontais (a transposta, Módulo 02). Antes, acrescentamos uma moldura de zeros em volta da imagem, o preenchimento (padding),
# para que o mapa tenha o mesmo tamanho $8 \times 8$ da imagem.

# %%
digitos = load_digits()
imagens = digitos.images / 16.0       # (1797, 8, 8), valores entre 0 e 1
rotulos = digitos.target

filtro_horizontal = filtro_vertical.T
digito = imagens[0]
com_moldura = np.pad(digito, 1)       # moldura de zeros: 10 x 10

fig, eixos = plt.subplots(1, 3, figsize=(11, 3.8))
for eixo, (titulo, mapa) in zip(eixos, [("dígito original", digito),
                                        ("filtro de bordas verticais", convoluir(com_moldura, filtro_vertical)),
                                        ("filtro de bordas horizontais", convoluir(com_moldura, filtro_horizontal))]):
    imagem = eixo.imshow(mapa, cmap="gray_r" if titulo == "dígito original" else "RdBu")
    eixo.set_title(titulo)
    eixo.axis("off")
    fig.colorbar(imagem, ax=eixo, fraction=0.046)
plt.show()

# %% [markdown]
# No mapa vertical, vermelho e azul marcam as bordas esquerda e direita dos traços do zero (onde a imagem fica mais clara ou mais escura
# andando para a direita). No horizontal, as bordas de cima e de baixo. Uma CNN não usa filtros feitos à mão: ela **aprende** os seus, pelo
# gradiente descendente, como qualquer outro peso.

# %% [markdown]
# ## 4. As outras peças: canais, pooling e camadas
#
# **Vários filtros, vários canais.** Uma camada convolucional usa vários filtros (por exemplo, 16), cada um gerando o seu mapa. O resultado
# é uma pilha de 16 mapas, chamados canais (channels). A camada seguinte tem filtros que olham **todos** os canais ao mesmo tempo: um filtro
# $3 \times 3$ sobre 16 canais tem $3 \times 3 \times 16 = 144$ pesos (mais um viés). Assim, camadas mais profundas combinam padrões simples
# (bordas) em padrões mais complexos (curvas, laços, cruzamentos).
#
# **Pooling.** Depois de uma convolução, é comum reduzir o mapa. O max pooling $2 \times 2$ divide o mapa em quadradinhos de $2 \times 2$ e
# guarda só o maior valor de cada um. Exemplo à mão: o quadradinho $\begin{pmatrix} 1 & 3 \\ 2 & 0 \end{pmatrix}$ vira $3$. O mapa fica com
# metade da altura e metade da largura. Vantagens: menos números para as próximas camadas, e um pouco de tolerância a deslocamentos (se o padrão
# andar um pixel dentro do quadradinho, o máximo continua o mesmo).
#
# **Ativação.** Como sempre, uma ReLU depois de cada convolução (Módulo 04).
#
# A nossa CNN para imagens $8 \times 8$:
#
# | camada | saída (canais × altura × largura) | parâmetros |
# |---|---|---|
# | entrada | 1 × 8 × 8 | 0 |
# | convolução 3×3, 16 filtros, padding 1, ReLU | 16 × 8 × 8 | $16 \cdot (9 + 1) = 160$ |
# | max pooling 2×2 | 16 × 4 × 4 | 0 |
# | convolução 3×3, 32 filtros, padding 1, ReLU | 32 × 4 × 4 | $32 \cdot (144 + 1) = 4640$ |
# | max pooling 2×2 | 32 × 2 × 2 | 0 |
# | achatar + camada densa | 10 | $128 \cdot 10 + 10 = 1290$ |
#
# Compare: a **primeira** camada densa do Módulo 05 (64 → 64) sozinha tem $64 \cdot 64 + 64 = 4160$ parâmetros. A primeira camada convolucional
# tem 160, porque os mesmos 10 números são usados em todas as posições da imagem.

# %% [markdown]
# ## 5. A convolução no PyTorch confere com a nossa
#
# No PyTorch, a camada é `nn.Conv2d(canais_entrada, canais_saida, tamanho_filtro, padding=...)`. Ela espera tensores com forma
# (lote, canais, altura, largura). Antes de usar, conferimos: colocamos o nosso filtro vertical dentro de uma camada e comparamos com a
# função `convoluir`.

# %%
camada = nn.Conv2d(1, 1, 3, padding=1, bias=False)
with torch.no_grad():
    camada.weight.copy_(torch.tensor(filtro_vertical).reshape(1, 1, 3, 3))
    mapa_pytorch = camada(torch.tensor(digito, dtype=torch.float32).reshape(1, 1, 8, 8))[0, 0].numpy()

mapa_nosso = convoluir(com_moldura, filtro_vertical)
print("maior diferença entre PyTorch e a nossa função:", np.abs(mapa_pytorch - mapa_nosso).max())

# %% [markdown]
# Os mesmos números (a diferença é só arredondamento do `float32`).

# %% [markdown]
# ## 6. Os dados e as duas redes
#
# A mesma divisão dos Módulos 05 e 07 (mesma semente): treino, validação e teste. As imagens entram com forma (lote, 1, 8, 8): um canal só,
# tons de cinza. Para comparar de forma justa, a rede densa do Módulo 07 recebe a mesma entrada e a achata com `nn.Flatten()`.

# %%
gerador_divisao = np.random.default_rng(42)
ordem = gerador_divisao.permutation(len(rotulos))
indices_treino, indices_validacao, indices_teste = ordem[:1078], ordem[1078:1437], ordem[1437:]


def para_tensores(indices, imagens_fonte=imagens):
    """Imagens com forma (n, 1, 8, 8) em float32 e rótulos inteiros."""
    return (torch.tensor(imagens_fonte[indices], dtype=torch.float32).unsqueeze(1),
            torch.tensor(rotulos[indices]))


X_treino, y_treino = para_tensores(indices_treino)
X_validacao, y_validacao = para_tensores(indices_validacao)
X_teste, y_teste = para_tensores(indices_teste)
print("treino:", tuple(X_treino.shape))


def criar_rede_densa():
    """A rede do Módulo 07: 64 -> 64 (ReLU) -> 10."""
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def criar_cnn():
    """A CNN da tabela da seção 4."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16 x 4 x 4
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32 x 2 x 2
        nn.Flatten(), nn.Linear(32 * 2 * 2, 10),
    )


def contar_parametros(rede):
    """Vimos isso no Módulo 07."""
    return sum(p.numel() for p in rede.parameters())


print(f"rede densa: {contar_parametros(criar_rede_densa())} parâmetros   CNN: {contar_parametros(criar_cnn())} parâmetros")

# %% [markdown]
# (`unsqueeze(1)` acrescenta o eixo dos canais: de (n, 8, 8) para (n, 1, 8, 8).) As duas redes têm tamanhos parecidos.

# %% [markdown]
# ## 7. Treinando as duas
#
# O laço de treino do Módulo 07: Adam com taxa 0,001, minilotes de 32, 40 épocas.

# %%
def acuracia(rede, X, y):
    """Vimos isso no Módulo 07."""
    with torch.no_grad():
        return (rede(X).argmax(dim=1) == y).float().mean().item()


def treinar(rede, X, y, epocas=40, tamanho_lote=32, taxa=0.001, semente=0):
    """Vimos isso no Módulo 07: Adam com minilotes. Guarda a acurácia de validação por época."""
    funcao_perda = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(rede.parameters(), lr=taxa)
    gerador = torch.Generator().manual_seed(semente)
    historico = []
    for epoca in range(epocas):
        ordem_lotes = torch.randperm(len(X), generator=gerador)
        for inicio in range(0, len(X), tamanho_lote):
            lote = ordem_lotes[inicio:inicio + tamanho_lote]
            perda = funcao_perda(rede(X[lote]), y[lote])
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
        historico.append(acuracia(rede, X_validacao, y_validacao))
    return historico


redes, historicos = {}, {}
for nome, criar in [("rede densa", criar_rede_densa), ("CNN", criar_cnn)]:
    torch.manual_seed(0)
    redes[nome] = criar()
    inicio = time.perf_counter()
    historicos[nome] = treinar(redes[nome], X_treino, y_treino)
    print(f"{nome:10s}: {time.perf_counter() - inicio:4.1f} s   acurácia de validação final {historicos[nome][-1]:.1%}")

# %% [markdown]
# **O aprendizado acontecendo**, época por época, na validação.

# %%
plt.figure(figsize=(7, 4))
for nome, historico in historicos.items():
    plt.plot(range(1, len(historico) + 1), historico, label=nome)
plt.ylim(0.8, 1.0)
plt.xlabel("época")
plt.ylabel("acurácia de validação")
plt.title("Rede densa contra CNN")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 8. O teste de verdade: dígitos fora do lugar
#
# As imagens do `load_digits` vêm bem centralizadas. Na vida real, um dígito pode estar um pouco para o lado. O que acontece se deslocarmos
# as imagens de teste **um pixel** para cima, para baixo, para a esquerda ou para a direita? (Numa imagem $8 \times 8$, um pixel é muito:
# 12% da largura.) Nenhuma das redes viu dígitos deslocados no treino.

# %%
def deslocar(imagens_originais, linhas, colunas):
    """Desloca as imagens; a faixa que "entra" pela borda é preenchida com zeros."""
    deslocadas = np.roll(imagens_originais, (linhas, colunas), axis=(1, 2))
    if linhas > 0:
        deslocadas[:, :linhas, :] = 0
    elif linhas < 0:
        deslocadas[:, linhas:, :] = 0
    if colunas > 0:
        deslocadas[:, :, :colunas] = 0
    elif colunas < 0:
        deslocadas[:, :, colunas:] = 0
    return deslocadas


direcoes = {"original": (0, 0), "para cima": (-1, 0), "para baixo": (1, 0), "para a esquerda": (0, -1), "para a direita": (0, 1)}

fig, eixos = plt.subplots(1, 5, figsize=(12, 2.8))
for eixo, (nome, (linhas, colunas)) in zip(eixos, direcoes.items()):
    eixo.imshow(deslocar(imagens[indices_teste[:1]], linhas, colunas)[0], cmap="gray_r")
    eixo.set_title(nome)
    eixo.axis("off")
plt.show()

resultados = {nome: [] for nome in redes}
for nome_direcao, (linhas, colunas) in direcoes.items():
    X_deslocado, _ = para_tensores(indices_teste, deslocar(imagens, linhas, colunas))
    for nome in redes:
        resultados[nome].append(acuracia(redes[nome], X_deslocado, y_teste))
    print(f"{nome_direcao:16s} rede densa {resultados['rede densa'][-1]:.1%}   CNN {resultados['CNN'][-1]:.1%}")

# %%
posicoes = np.arange(len(direcoes))
plt.figure(figsize=(9, 4))
plt.bar(posicoes - 0.2, resultados["rede densa"], width=0.4, label="rede densa")
plt.bar(posicoes + 0.2, resultados["CNN"], width=0.4, label="CNN")
plt.xticks(posicoes, list(direcoes.keys()))
plt.ylabel("acurácia no teste")
plt.ylim(0, 1)
plt.title("Dígitos deslocados um pixel")
plt.legend()
plt.grid(True, axis="y")
plt.show()

# %% [markdown]
# Com as imagens originais, as duas redes vão bem. Deslocadas um pixel, **as duas** sofrem muito: numa imagem $8 \times 8$, um pixel é bastante,
# e nenhuma delas viu dígitos fora do lugar no treino. Mas a CNN resiste melhor em todas as direções, e muito melhor em algumas.
#
# A rede densa aprendeu algo como "tinta na entrada 28 e na entrada 36"; quando a tinta muda de entrada, ela fica perdida. Os filtros da CNN
# procuram os padrões em **todas** as posições, e o pooling tolera pequenos desvios. A tolerância da nossa CNN, porém, é parcial: no fim dela há
# uma camada densa que olha a **posição** dos mapas $2 \times 2$, e dois poolings dão só uma folga pequena. CNNs maiores, com mais camadas e um
# pooling sobre o mapa inteiro no fim (global pooling), chegam bem mais perto da invariância à translação (translation invariance), a
# resistência a deslocamentos que fez as CNNs dominarem a visão computacional por mais de uma década.

# %% [markdown]
# ## 9. O que a CNN aprendeu
#
# Os 16 filtros da primeira camada, e os mapas que eles produzem para um dígito. Azul e vermelho são pesos (ou ativações) de sinais opostos.

# %%
filtros = redes["CNN"][0].weight.detach()[:, 0].numpy()           # (16, 3, 3)
fig, eixos = plt.subplots(2, 8, figsize=(12, 3.4))
for i, eixo in enumerate(eixos.ravel()):
    limite = np.abs(filtros[i]).max()
    eixo.imshow(filtros[i], cmap="RdBu", vmin=-limite, vmax=limite)
    eixo.axis("off")
fig.suptitle("Os 16 filtros 3×3 aprendidos pela primeira camada")
plt.show()

with torch.no_grad():
    mapas = torch.relu(redes["CNN"][0](X_teste[:1]))[0].numpy()    # (16, 8, 8), depois da ReLU
fig, eixos = plt.subplots(2, 9, figsize=(13, 3.2))
eixos = eixos.ravel()
eixos[0].imshow(X_teste[0, 0].numpy(), cmap="gray_r")
eixos[0].set_title(f"dígito {y_teste[0].item()}")
for i in range(16):
    eixos[i + 1].imshow(mapas[i], cmap="viridis")
for eixo in eixos:
    eixo.axis("off")
fig.suptitle("Os 16 mapas de ativação da primeira camada para um dígito")
plt.show()

# %% [markdown]
# Com imagens tão pequenas e pouco treino, os filtros saem "ruidosos", bem menos limpos que os nossos. Ainda assim, vários têm uma parte
# positiva e outra negativa lado a lado, o formato de um detector de borda (como os da seção 3), em direções variadas. Cada mapa "acende" numa
# parte diferente do dígito, e as camadas seguintes combinam essas peças. Em redes grandes, treinadas com milhões de fotos, os filtros da
# primeira camada ficam nítidos: bordas, listras e manchas de cor.
#
# **Um pouco de história.** A ideia vem dos anos 1980 e 1990: Yann LeCun e colegas usaram CNNs (a LeNet) para ler códigos postais e cheques.
# Em 2012, uma CNN bem maior, a AlexNet, treinada em GPUs com mais de um milhão de fotos, venceu a competição ImageNet com folga e deu início à
# onda atual de aprendizado profundo. Imagens reais são grandes ($224 \times 224$ pixels, 3 canais de cor) e as redes têm dezenas de camadas,
# mas as peças são exatamente as deste módulo.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Calcule à mão a convolução da imagem $3 \times 4$ abaixo com o filtro de bordas horizontais
# $K = \begin{pmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{pmatrix}$ (o resultado tem $1 \times 2$ valores). Depois aplique max pooling $2 \times 2$,
# à mão, na matriz $4 \times 4$ abaixo. Confira com `convoluir` e com `nn.MaxPool2d(2)`.
#
# $$ \text{imagem} = \begin{pmatrix} 0&0&0&0 \\ 0&0&0&0 \\ 1&1&1&1 \end{pmatrix} \qquad \text{mapa} = \begin{pmatrix} 1&5&2&0 \\ 3&2&1&4 \\ 0&0&6&1 \\ 2&1&3&3 \end{pmatrix} $$
#
# **Exercício 2 (médio).** Imagine a mesma CNN **sem** padding (`padding=0`). Calcule à mão o tamanho da saída de cada camada para uma imagem
# $8 \times 8$ (lembre: um filtro $3 \times 3$ sem padding tira 2 da altura e da largura; o pooling divide por 2, descartando a sobra) e o número
# de entradas da camada densa final. Confira passando um tensor de zeros com forma (1, 1, 8, 8) por cada camada e imprimindo as formas.
#
# **Exercício 3 (desafio).** Aumento de dados (data augmentation): treine a **rede densa** de novo, mas, em cada época, desloque cada imagem de
# treino um pixel numa direção sorteada (ou não desloque). Depois meça a acurácia nas cinco versões do teste da seção 8. A rede densa fica mais
# resistente? Ela alcança a CNN?

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | convolutional neural network | rede neural convolucional | *A convolutional neural network slides small filters over the image.* |
# | convolution | convolução | *A convolution multiplies and adds inside a small window.* |
# | filter / kernel | filtro / núcleo | *This filter detects vertical edges.* |
# | feature map | mapa de ativação | *The feature map shows where the pattern appears.* |
# | edge detection | detecção de bordas | *Edge detection looks for fast changes in brightness.* |
# | padding | preenchimento | *Padding keeps the output the same size as the input.* |
# | stride | passo do filtro | *With a stride of two, the filter jumps two pixels at a time.* |
# | channel | canal | *Color images have three channels.* |
# | pooling | pooling (redução) | *Max pooling keeps the largest value in each window.* |
# | weight sharing | compartilhamento de pesos | *Weight sharing uses the same filter everywhere.* |
# | translation invariance | invariância à translação | *Translation invariance makes the CNN robust to shifts.* |
# | data augmentation | aumento de dados | *Data augmentation creates new training images by shifting them.* |
# | computer vision | visão computacional | *CNNs dominated computer vision for a decade.* |

# %% [markdown]
# ## Resumo
#
# - Uma convolução desliza um pequeno filtro pela imagem, fazendo um produto escalar em cada posição; o resultado é um mapa de ativação.
# - Um filtro de bordas é uma derivada discreta: ele acende onde a imagem muda depressa.
# - Canais, ReLU e pooling, empilhados, formam uma CNN, que usa os mesmos pesos em toda a imagem e por isso tem poucos parâmetros.
# - Com dígitos centralizados, CNN e rede densa ficam próximas; deslocados, as duas sofrem, mas a CNN resiste melhor (invariância à translação,
#   parcial nesta rede pequena).
# - Os filtros aprendidos lembram detectores de borda; as mesmas peças, em escala muito maior, levaram à revolução da visão computacional.
#
# **Próximo passo:** o bônus B3, um projeto escolhido por você, se quiser continuar.

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
# Convolução: nas duas posições, a linha de cima do filtro ($-1$) cai sobre zeros, a do meio ($0$) sobre zeros, e a de baixo ($+1$) sobre três uns:
# o resultado é $(3, 3)$. Pooling: os quadradinhos são $\{1, 5, 3, 2\}$, $\{2, 0, 1, 4\}$, $\{0, 0, 2, 1\}$ e $\{6, 1, 3, 3\}$; os máximos formam
# $\begin{pmatrix} 5 & 4 \\ 2 & 6 \end{pmatrix}$.

# %%
imagem_ex1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)
print(convoluir(imagem_ex1, filtro_horizontal))

mapa_ex1 = torch.tensor([[1, 5, 2, 0], [3, 2, 1, 4], [0, 0, 6, 1], [2, 1, 3, 3]], dtype=torch.float32)
print(nn.MaxPool2d(2)(mapa_ex1.reshape(1, 1, 4, 4))[0, 0])

# %% [markdown]
# ### Solução do Exercício 2
#
# $8 \times 8$ → convolução sem padding: $6 \times 6$ (16 canais) → pooling: $3 \times 3$ → convolução: $1 \times 1$ (32 canais) → pooling de um mapa
# $1 \times 1$: não cabe nenhum quadradinho $2 \times 2$, e o resultado seria vazio! Sem padding, a imagem encolhe rápido demais. Por isso, para imagens
# pequenas, o padding é essencial. Removendo o segundo pooling, a camada densa receberia $32 \times 1 \times 1 = 32$ entradas.

# %%
camadas_sem_padding = [nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3), nn.ReLU()]
x = torch.zeros(1, 1, 8, 8)
print("entrada:", tuple(x.shape))
for camada_ex2 in camadas_sem_padding:
    x = camada_ex2(x)
    print(f"{camada_ex2.__class__.__name__:10s} -> {tuple(x.shape)}")
try:
    nn.MaxPool2d(2)(x)
except RuntimeError:
    print("MaxPool2d     -> erro: o mapa 1 x 1 é pequeno demais para um pooling 2 x 2")

# %% [markdown]
# ### Solução do Exercício 3

# %%
deslocamentos_possiveis = list(direcoes.values())          # inclui (0, 0): "não deslocar"
gerador_aumento = np.random.default_rng(0)
imagens_treino = imagens[indices_treino]

torch.manual_seed(0)
rede_aumentada = criar_rede_densa()
funcao_perda = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(rede_aumentada.parameters(), lr=0.001)
for epoca in range(40):
    # uma versão deslocada nova das imagens de treino a cada época
    escolhas = gerador_aumento.integers(0, len(deslocamentos_possiveis), len(imagens_treino))
    aumentadas = np.zeros_like(imagens_treino)
    for k, (linhas, colunas) in enumerate(deslocamentos_possiveis):
        selecionadas = escolhas == k
        aumentadas[selecionadas] = deslocar(imagens_treino[selecionadas], linhas, colunas)
    X_aumentado = torch.tensor(aumentadas, dtype=torch.float32).unsqueeze(1)
    ordem_lotes = torch.randperm(len(X_aumentado))
    for inicio in range(0, len(X_aumentado), 32):
        lote = ordem_lotes[inicio:inicio + 32]
        perda = funcao_perda(rede_aumentada(X_aumentado[lote]), y_treino[lote])
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

for (nome_direcao, (linhas, colunas)), densa, cnn in zip(direcoes.items(), resultados["rede densa"], resultados["CNN"]):
    X_deslocado, _ = para_tensores(indices_teste, deslocar(imagens, linhas, colunas))
    print(f"{nome_direcao:16s} densa {densa:.1%}   densa com aumento {acuracia(rede_aumentada, X_deslocado, y_teste):.1%}   CNN {cnn:.1%}")

# %% [markdown]
# Com o aumento de dados, a rede densa fica muito mais resistente aos deslocamentos e, aqui, chega a superar a CNN treinada sem aumento nas
# imagens deslocadas (perdendo um pouco nas originais). Ela viu dígitos fora do lugar no treino e aprendeu a reconhecê-los. É uma técnica muito
# usada, também com CNNs (girar, espelhar, mudar o brilho das fotos), e uma CNN com aumento de dados seria ainda melhor. A diferença é de
# filosofia: a rede densa precisa **ver** cada deslocamento nos dados; a CNN traz parte da tolerância na própria estrutura.
