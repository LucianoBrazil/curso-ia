# %% [markdown]
# # Módulo 05 — Reconhecendo dígitos escritos à mão
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/05-digitos.ipynb)
#
# **Objetivo.** Levar a rede do Módulo 04 a um problema de verdade: reconhecer os algarismos de 0 a 9 em pequenas imagens.
# Para isso você vai aprender a softmax (escolher entre 10 classes), a separar os dados em treino, validação e teste,
# a treinar em minilotes e a perceber o sobreajuste.
#
# **O que você vai construir.** Uma rede neural, feita só com NumPy, que acerta mais de 95% dos dígitos que nunca viu,
# e uma galeria dos dígitos em que ela erra.

# %% [markdown]
# ## 1. A intuição: uma imagem é uma tabela de números
#
# Uma foto em tons de cinza é uma grade de pontinhos, os pixels (pixels). Cada pixel guarda um número: a intensidade
# do cinza. Para o computador, a imagem **é** essa tabela.
#
# O conjunto de dados `load_digits`, que vem dentro do scikit-learn, tem 1797 dígitos escritos à mão por pessoas diferentes.
# Cada imagem tem $8 \times 8 = 64$ pixels, com intensidade de 0 (branco) a 16 (preto).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digitos = load_digits()
imagens = digitos.images        # forma (1797, 8, 8)
rotulos = digitos.target        # o algarismo de cada imagem, de 0 a 9
print("imagens:", imagens.shape, "  rótulos:", rotulos.shape)

fig, eixos = plt.subplots(2, 10, figsize=(12, 3))
for i, eixo in enumerate(eixos.ravel()):
    eixo.imshow(imagens[i], cmap="gray_r")
    eixo.set_title(str(rotulos[i]))
    eixo.axis("off")
plt.show()

# %% [markdown]
# A mesma imagem, como números. Dá para "ver" o zero na tabela:

# %%
print(imagens[0].astype(int))

# %% [markdown]
# A rede do Módulo 04 recebe um vetor de entradas, não uma tabela. Então "achatamos" cada imagem: as 8 linhas de 8 pixels
# viram um vetor de 64 números. O conjunto inteiro vira uma matriz $X$ de forma $1797 \times 64$, uma linha por imagem
# (o scikit-learn já entrega isso pronto em `digitos.data`).
#
# Dividimos por 16 para que as entradas fiquem entre 0 e 1. É uma normalização mais simples que a do Módulo 02, suficiente
# aqui porque todos os pixels já estão na mesma escala.

# %%
X_todos = digitos.data / 16.0
print("forma de X:", X_todos.shape, "  menor:", X_todos.min(), "  maior:", X_todos.max())

# %% [markdown]
# ## 2. Treino, validação e teste
#
# No Módulo 03 medimos a acurácia nos mesmos dados usados para treinar. Isso é como corrigir uma prova com as questões
# que o aluno já decorou: a nota engana. O que interessa é o desempenho em dados **novos**, a generalização (generalization).
#
# Por isso separamos os dados em três partes, sorteadas:
#
# - **treino (training set), 60%:** o material de estudo. A rede ajusta os pesos olhando só para ele.
# - **validação (validation set), 20%:** os simulados. Usamos para comparar escolhas (taxa de aprendizado, tamanho da rede,
#   quando parar), sem treinar neles.
# - **teste (test set), 20%:** a prova final. Usada **uma única vez**, no fim. Se olharmos para ela várias vezes para fazer
#   escolhas, ela vira mais um simulado e deixa de medir a generalização de forma honesta.

# %%
rng = np.random.default_rng(42)
n_total = len(rotulos)
ordem = rng.permutation(n_total)          # os índices de 0 a 1796, embaralhados

n_treino = int(0.6 * n_total)
n_validacao = int(0.2 * n_total)
indices_treino = ordem[:n_treino]
indices_validacao = ordem[n_treino:n_treino + n_validacao]
indices_teste = ordem[n_treino + n_validacao:]

X_treino, rotulos_treino = X_todos[indices_treino], rotulos[indices_treino]
X_validacao, rotulos_validacao = X_todos[indices_validacao], rotulos[indices_validacao]
X_teste, rotulos_teste = X_todos[indices_teste], rotulos[indices_teste]
print(f"treino: {len(X_treino)}   validação: {len(X_validacao)}   teste: {len(X_teste)}")

# %% [markdown]
# ## 3. Dez classes: a softmax
#
# No Módulo 04 a rede tinha **uma** saída: a probabilidade de a lâmpada acender. Agora há 10 respostas possíveis.
# A rede terá 10 saídas, uma pontuação $z_k$ para cada algarismo $k$. Precisamos transformar as 10 pontuações em
# 10 probabilidades: todas positivas e somando 1.
#
# A softmax faz isso em dois passos: torna cada pontuação positiva com a exponencial e depois divide pelo total.
#
# $$ p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}} $$
#
# **Exemplo à mão**, com só 3 classes e pontuações $z = (2, 1, 0)$:
#
# - exponenciais: $e^2 \approx 7{,}389$, $\; e^1 \approx 2{,}718$, $\; e^0 = 1$; total $\approx 11{,}107$;
# - probabilidades: $7{,}389/11{,}107 \approx 0{,}665$, $\; 2{,}718/11{,}107 \approx 0{,}245$, $\; 1/11{,}107 \approx 0{,}090$.
#
# A maior pontuação fica com a maior probabilidade, mas as outras não zeram. Com duas classes, a softmax é a sigmoide
# do Módulo 03 disfarçada.
#
# **Um cuidado prático.** $e^{1000}$ estoura a memória do computador. Como somar a mesma constante a todas as pontuações
# não muda a softmax (ela aparece em cima e embaixo da fração e se cancela), subtraímos a maior pontuação antes da exponencial.

# %%
def softmax(Z):
    """Transforma cada linha de pontuações em probabilidades (positivas, somando 1)."""
    Z_estavel = Z - Z.max(axis=1, keepdims=True)
    exponenciais = np.exp(Z_estavel)
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


print(np.round(softmax(np.array([[2.0, 1.0, 0.0]])), 3))
print(np.round(softmax(np.array([[1002.0, 1001.0, 1000.0]])), 3), "(sem estourar)")

# %% [markdown]
# O `keepdims=True` mantém o resultado como coluna, para que cada linha seja dividida pelo **seu** total.

# %% [markdown]
# ## 4. Rótulos como vetores e a entropia cruzada com várias classes
#
# A rede devolve um vetor de 10 probabilidades. Para comparar, escrevemos o rótulo também como vetor: tudo zero, com um 1
# na posição da classe certa. O dígito 3 vira $(0, 0, 0, 1, 0, 0, 0, 0, 0, 0)$. Isso se chama codificação one-hot
# (one-hot encoding).
#
# A entropia cruzada do Módulo 03 vira
#
# $$ L = -\sum_{k} y_k \log p_k = -\log p_{\text{classe certa}} $$
#
# porque só o termo da classe certa tem $y_k = 1$. A ideia é a mesma: punir pouco quando a rede dá probabilidade alta para
# a classe certa e muito quando dá probabilidade baixa. No exemplo à mão, se a classe certa fosse a primeira,
# $L = -\log 0{,}665 \approx 0{,}408$.
#
# **A boa notícia:** o gradiente da perda em relação às pontuações continua sendo
#
# $$ \frac{\partial L}{\partial z_k} = p_k - y_k $$
#
# exatamente como no Módulo 03. A dedução é parecida com aquela, só que com mais termos; em vez de repeti-la, vamos
# conferir numericamente na seção 6, como no Módulo 04.

# %%
def codificar_one_hot(rotulos, n_classes=10):
    """Rótulo inteiro -> vetor com 1 na posição da classe e 0 no resto."""
    Y = np.zeros((len(rotulos), n_classes))
    Y[np.arange(len(rotulos)), rotulos] = 1
    return Y


def entropia_cruzada(P, Y):
    """Entropia cruzada média para várias classes. O clip evita log(0)."""
    return -np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1))


Y_treino = codificar_one_hot(rotulos_treino)
Y_validacao = codificar_one_hot(rotulos_validacao)
Y_teste = codificar_one_hot(rotulos_teste)
print("rótulo:", rotulos_treino[0], "  one-hot:", Y_treino[0])
print(f"perda do exemplo à mão: {entropia_cruzada(softmax(np.array([[2.0, 1.0, 0.0]])), np.array([[1.0, 0, 0]])):.3f}")

# %% [markdown]
# ## 5. A rede: 64 entradas, 64 neurônios escondidos, 10 saídas
#
# É a rede do Módulo 04 com duas mudanças: 10 saídas com softmax, em vez de uma com sigmoide, e uma inicialização mais
# cuidadosa dos pesos.
#
# **Por que a inicialização importa agora.** Cada neurônio escondido soma 64 entradas. Se cada peso tiver tamanho típico 1,
# a soma fica grande demais, e a camada seguinte, que soma 64 desses valores, fica maior ainda. As saídas explodem e a
# softmax fica "travada" em 0 e 1 desde o começo. A solução é sortear os pesos com desvio padrão $\sqrt{2/n_{\text{entradas}}}$.
# Essa receita, pensada para a ReLU, se chama inicialização de He (He initialization).
#
# Funções `relu`, `derivada_relu` e a estrutura da rede: vimos isso no Módulo 04.

# %%
def relu(z):
    """Vimos isso no Módulo 04."""
    return np.maximum(0, z)


def derivada_relu(z):
    """Vimos isso no Módulo 04."""
    return (z > 0).astype(float)


def inicializar_rede(n_entradas, n_ocultos, n_saidas, semente=0):
    """Pesos sorteados com a inicialização de He; vieses zerados."""
    gerador = np.random.default_rng(semente)
    return {
        "W1": gerador.normal(0, np.sqrt(2 / n_entradas), (n_entradas, n_ocultos)),
        "b1": np.zeros(n_ocultos),
        "W2": gerador.normal(0, np.sqrt(2 / n_ocultos), (n_ocultos, n_saidas)),
        "b2": np.zeros(n_saidas),
    }


def propagacao_frente(X, rede):
    """Como no Módulo 04, mas com softmax na saída."""
    Z1 = X @ rede["W1"] + rede["b1"]
    H = relu(Z1)
    P = softmax(H @ rede["W2"] + rede["b2"])
    return P, {"X": X, "Z1": Z1, "H": H, "P": P}


def retropropagacao(Y, rede, memoria):
    """Idêntica à do Módulo 04: o primeiro elo continua sendo P - Y."""
    n = len(Y)
    dZ2 = (memoria["P"] - Y) / n
    dW2 = memoria["H"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dH = dZ2 @ rede["W2"].T
    dZ1 = dH * derivada_relu(memoria["Z1"])
    dW1 = memoria["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

# %% [markdown]
# ## 6. Conferindo os gradientes
#
# Mudamos a saída, então conferimos de novo, como no Módulo 04. Para ser rápido, usamos uma rede pequena
# (5 neurônios escondidos) e só 20 imagens.

# %%
def conferir_gradientes(X, Y, rede, passinho=1e-5):
    """Vimos isso no Módulo 04: compara a retropropagação com o passinho para cada lado."""
    analiticos = retropropagacao(Y, rede, propagacao_frente(X, rede)[1])
    for nome in rede:
        numericos = np.zeros_like(rede[nome])
        for posicao in np.ndindex(rede[nome].shape):
            original = rede[nome][posicao]
            rede[nome][posicao] = original + passinho
            perda_mais = entropia_cruzada(propagacao_frente(X, rede)[0], Y)
            rede[nome][posicao] = original - passinho
            perda_menos = entropia_cruzada(propagacao_frente(X, rede)[0], Y)
            rede[nome][posicao] = original
            numericos[posicao] = (perda_mais - perda_menos) / (2 * passinho)
        print(f"{nome}: maior diferença = {np.max(np.abs(numericos - analiticos[nome])):.2e}")


conferir_gradientes(X_treino[:20], Y_treino[:20], inicializar_rede(64, 5, 10))

# %% [markdown]
# Diferenças minúsculas: a fórmula $p - y$ vale mesmo para a softmax.

# %% [markdown]
# ## 7. Minilotes e épocas
#
# Até agora, cada passo do gradiente descendente olhava **todos** os exemplos. Com milhões de exemplos, cada passo
# demoraria demais. A alternativa é usar, a cada passo, só um pedaço sorteado dos dados: um minilote (mini-batch),
# por exemplo de 32 imagens.
#
# - O gradiente de um minilote é uma **estimativa** do gradiente verdadeiro: aponta mais ou menos para o mesmo lado,
#   com algum ruído. É como uma pesquisa eleitoral: 32 entrevistas já dão uma boa ideia da opinião de todos.
# - Em vez de 1 passo caro, damos muitos passos baratos. O ruído até ajuda a escapar de lugares ruins da paisagem do erro.
# - Uma época (epoch) é uma passada completa pelos dados de treino. Com 1078 imagens e lotes de 32, uma época tem 34 passos.
#
# Esse método é o gradiente descendente estocástico (stochastic gradient descent, SGD). "Estocástico" quer dizer "com sorteio".
# A cada época, embaralhamos a ordem dos exemplos, para que os lotes sejam diferentes.

# %%
def avaliar(X, Y, rede):
    """Perda e acurácia da rede num conjunto de dados."""
    P = propagacao_frente(X, rede)[0]
    return entropia_cruzada(P, Y), np.mean(P.argmax(axis=1) == Y.argmax(axis=1))


def treinar_minilotes(X, Y, X_val, Y_val, n_ocultos, taxa, epocas, tamanho_lote, semente=0):
    """SGD com minilotes. Mede perda e acurácia no treino e na validação ao fim de cada época."""
    rede = inicializar_rede(X.shape[1], n_ocultos, Y.shape[1], semente)
    gerador = np.random.default_rng(semente)
    historico = {"perda_treino": [], "perda_val": [], "acc_treino": [], "acc_val": []}
    for epoca in range(epocas):
        ordem = gerador.permutation(len(X))
        for inicio in range(0, len(X), tamanho_lote):
            lote = ordem[inicio:inicio + tamanho_lote]
            P, memoria = propagacao_frente(X[lote], rede)
            gradientes = retropropagacao(Y[lote], rede, memoria)
            for nome in rede:
                rede[nome] = rede[nome] - taxa * gradientes[nome]
        perda_treino, acc_treino = avaliar(X, Y, rede)
        perda_val, acc_val = avaliar(X_val, Y_val, rede)
        historico["perda_treino"].append(perda_treino)
        historico["acc_treino"].append(acc_treino)
        historico["perda_val"].append(perda_val)
        historico["acc_val"].append(acc_val)
    return rede, historico

# %% [markdown]
# A previsão é a classe de maior probabilidade: `P.argmax(axis=1)` devolve, para cada linha, a posição do maior valor.
#
# ## 8. Treinando
#
# 64 neurônios escondidos, taxa 0,1, lotes de 32, 40 épocas.

# %%
rede, historico = treinar_minilotes(X_treino, Y_treino, X_validacao, Y_validacao,
                                    n_ocultos=64, taxa=0.1, epocas=40, tamanho_lote=32)
print(f"acurácia no treino: {historico['acc_treino'][-1]:.1%}   na validação: {historico['acc_val'][-1]:.1%}")

# %% [markdown]
# **O aprendizado acontecendo**, época por época, no treino e na validação.

# %%
def desenhar_historico(historico, titulo):
    """Perda e acurácia por época, no treino e na validação."""
    fig, (eixo_perda, eixo_acc) = plt.subplots(1, 2, figsize=(12, 4))
    eixo_perda.plot(historico["perda_treino"], label="treino")
    eixo_perda.plot(historico["perda_val"], label="validação")
    eixo_perda.set_xlabel("época")
    eixo_perda.set_ylabel("entropia cruzada")
    eixo_perda.set_yscale("log")
    eixo_perda.legend()
    eixo_perda.grid(True)
    eixo_acc.plot(historico["acc_treino"], label="treino")
    eixo_acc.plot(historico["acc_val"], label="validação")
    eixo_acc.set_xlabel("época")
    eixo_acc.set_ylabel("acurácia")
    eixo_acc.legend()
    eixo_acc.grid(True)
    fig.suptitle(titulo)
    plt.show()


desenhar_historico(historico, "Treino com 1078 imagens")

# %% [markdown]
# As curvas de treino e validação andam juntas. A validação fica um pouco atrás: é natural ir um pouco pior em dados novos.

# %% [markdown]
# ## 9. Sobreajuste: decorar em vez de aprender
#
# Um aluno pode tirar 10 em todos os exercícios da apostila porque **decorou as respostas**, e ir mal na prova.
# Uma rede pode fazer o mesmo: com poucos dados e muitos pesos, ela decora os exemplos de treino, inclusive os detalhes
# que são puro acaso, e generaliza mal. Isso é o sobreajuste (overfitting).
#
# Como perceber: a perda de treino continua caindo enquanto a de **validação para de cair e começa a subir**.
# A distância entre as duas curvas é o sinal de alerta.
#
# Para provocar o sobreajuste, vamos treinar a mesma rede com só 100 imagens, por muitas épocas e com uma taxa maior (0,5).

# %%
rede_pequena, historico_pequeno = treinar_minilotes(X_treino[:100], Y_treino[:100], X_validacao, Y_validacao,
                                                    n_ocultos=64, taxa=0.5, epocas=300, tamanho_lote=32)
melhor_epoca = int(np.argmin(historico_pequeno["perda_val"]))
print(f"acurácia no treino: {historico_pequeno['acc_treino'][-1]:.1%}   na validação: {historico_pequeno['acc_val'][-1]:.1%}")
print(f"a perda de validação foi mínima na época {melhor_epoca}")

desenhar_historico(historico_pequeno, "Treino com só 100 imagens: sobreajuste")

# %% [markdown]
# A rede acerta 100% das 100 imagens de treino, e a perda de treino continua despencando. Na validação, a acurácia fica bem
# abaixo, e a perda para de cair por volta da época 100; depois fica parada e chega a subir um pouco. Treinar mais não ajuda: a rede só
# fica mais **confiante** nas respostas que decorou. A distância enorme entre as curvas de treino e de validação é o sinal
# mais claro do sobreajuste. Remédios comuns:
#
# - **mais dados** (a curva da seção 8, com 1078 imagens, quase não se afastou);
# - **rede menor**, com menos pesos para decorar;
# - **parada antecipada (early stopping):** guardar a rede da época com menor perda de validação (o Exercício 3 faz isso);
# - **regularização (regularization):** técnicas que penalizam pesos grandes, que veremos rapidamente mais adiante no curso.

# %% [markdown]
# ## 10. A prova final: o conjunto de teste
#
# Agora, e só agora, usamos o teste, com a rede da seção 8.

# %%
perda_teste, acuracia_teste = avaliar(X_teste, Y_teste, rede)
print(f"acurácia no teste: {acuracia_teste:.1%}")

# %% [markdown]
# A matriz de confusão do Módulo 03, agora $10 \times 10$: a linha é o dígito real, a coluna é o previsto.
# A diagonal são os acertos; o resto mostra quais dígitos a rede confunde.

# %%
previstos_teste = propagacao_frente(X_teste, rede)[0].argmax(axis=1)
matriz = np.zeros((10, 10), dtype=int)
for real, previsto in zip(rotulos_teste, previstos_teste):
    matriz[real, previsto] += 1

fig, eixo = plt.subplots(figsize=(6.5, 6))
eixo.imshow(matriz, cmap="Blues")
for real in range(10):
    for previsto in range(10):
        if matriz[real, previsto] > 0:
            cor = "white" if matriz[real, previsto] > matriz.max() / 2 else "black"
            eixo.text(previsto, real, matriz[real, previsto], ha="center", va="center", color=cor)
eixo.set_xticks(range(10))
eixo.set_yticks(range(10))
eixo.set_xlabel("previsto")
eixo.set_ylabel("real")
eixo.set_title(f"Matriz de confusão no teste ({acuracia_teste:.1%})")
plt.show()

# %% [markdown]
# ## 11. Onde a rede erra
#
# Olhar os erros ensina muito. Abaixo, os dígitos do teste que a rede errou, com a resposta certa, o palpite e a confiança
# do palpite. Vários são difíceis até para nós.

# %%
P_teste = propagacao_frente(X_teste, rede)[0]
erros = np.where(previstos_teste != rotulos_teste)[0]
print(f"{len(erros)} erros em {len(rotulos_teste)} imagens de teste")

n_mostrar = min(len(erros), 12)
fig, eixos = plt.subplots(2, 6, figsize=(12, 5.5))
for eixo in eixos.ravel():
    eixo.axis("off")
for eixo, indice in zip(eixos.ravel(), erros[:n_mostrar]):
    eixo.imshow(X_teste[indice].reshape(8, 8), cmap="gray_r")
    eixo.set_title(f"real {rotulos_teste[indice]}, previu {previstos_teste[indice]}\n({P_teste[indice].max():.0%})", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# `np.where(condição)[0]` devolve as posições em que a condição é verdadeira. `reshape(8, 8)` desfaz o "achatamento".

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Calcule à mão a softmax de $z = (3, 1, 1)$, usando $e^3 \approx 20{,}09$ e $e \approx 2{,}718$.
# Depois calcule a de $z = (103, 101, 101)$. Por que dá o mesmo resultado? Confira com a função `softmax`.
#
# **Exercício 2 (médio).** Compare três tamanhos de lote: 1, 32 e 1078 (o treino inteiro de uma vez), com taxa 0,1 e 20 épocas.
# Desenhe a acurácia de validação por época para os três e meça o tempo de cada treino com `time.perf_counter()`.
# Qual aprende mais por época? Qual é mais rápido por época?
#
# **Exercício 3 (desafio).** Implemente a parada antecipada. Treine com as 100 imagens da seção 9 (taxa 0,5, 300 épocas) e guarde
# uma cópia da rede sempre que a perda de validação atingir um novo mínimo. No fim, compare a acurácia **de validação**
# da rede guardada com a da rede final. (Dica: `{nome: valor.copy() for nome, valor in rede.items()}` copia o dicionário
# inteiro. Sem `.copy()`, a "cópia" mudaria junto com a rede.)

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | pixel | pixel | *Each image has 64 pixels.* |
# | softmax | softmax | *Softmax turns scores into probabilities.* |
# | one-hot encoding | codificação one-hot | *The digit 3 becomes a one-hot vector.* |
# | training set | conjunto de treino | *The network only learns from the training set.* |
# | validation set | conjunto de validação | *We use the validation set to choose the learning rate.* |
# | test set | conjunto de teste | *Look at the test set only once, at the end.* |
# | generalization | generalização | *Good generalization means doing well on new data.* |
# | mini-batch | minilote | *Each step uses a mini-batch of 32 images.* |
# | epoch | época | *One epoch is one pass over the training data.* |
# | stochastic gradient descent | gradiente descendente estocástico | *Stochastic gradient descent uses random mini-batches.* |
# | overfitting | sobreajuste | *Overfitting is memorizing instead of learning.* |
# | early stopping | parada antecipada | *Early stopping keeps the best model on validation.* |
# | initialization | inicialização | *He initialization keeps the signals at a good scale.* |
# | misclassified | classificado errado | *Some misclassified digits are hard even for humans.* |

# %% [markdown]
# ## Resumo
#
# - Uma imagem em tons de cinza é uma tabela de números; achatada, vira um vetor de entradas para a rede.
# - A softmax transforma 10 pontuações em 10 probabilidades; com one-hot e entropia cruzada, o gradiente continua $p - y$.
# - Treino para aprender, validação para escolher, teste uma única vez para medir a generalização.
# - Minilotes dão muitos passos baratos e ruidosos; uma época é uma passada completa pelo treino.
# - Sobreajuste aparece quando a perda de validação sobe enquanto a de treino cai; mais dados e parada antecipada ajudam.
#
# **Próximo módulo:** Física + IA. A rede vai aprender a prever o alcance de um projétil com resistência do ar.

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
# Exponenciais: $20{,}09$, $2{,}718$ e $2{,}718$; total $25{,}52$. Probabilidades: $20{,}09/25{,}52 \approx 0{,}787$ e
# $2{,}718/25{,}52 \approx 0{,}107$ (duas vezes).
#
# Para $z = (103, 101, 101)$, todas as pontuações ganharam $+100$. Isso multiplica cada exponencial por $e^{100}$, em cima e
# embaixo da fração, e o fator se cancela. A softmax só se importa com as **diferenças** entre as pontuações.

# %%
print(np.round(softmax(np.array([[3.0, 1.0, 1.0], [103.0, 101.0, 101.0]])), 3))

# %% [markdown]
# ### Solução do Exercício 2

# %%
import time

plt.figure(figsize=(7, 4))
for tamanho_lote in [1, 32, len(X_treino)]:
    inicio = time.perf_counter()
    _, hist = treinar_minilotes(X_treino, Y_treino, X_validacao, Y_validacao,
                                n_ocultos=64, taxa=0.1, epocas=20, tamanho_lote=tamanho_lote)
    duracao = time.perf_counter() - inicio
    print(f"lote {tamanho_lote:5d}: {duracao:5.2f} s   acurácia final de validação {hist['acc_val'][-1]:.1%}")
    plt.plot(hist["acc_val"], label=f"lote {tamanho_lote}")
plt.xlabel("época")
plt.ylabel("acurácia de validação")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Com o treino inteiro em cada passo, há só 1 passo por época: a rede aprende muito pouco em 20 épocas, embora cada época seja
# rapidíssima. Com lote 1, há 1078 passos por época: aprende rápido por época, mas cada época é lenta, porque o NumPy faz
# milhares de contas pequenas em vez de poucas contas grandes (a vetorização do Módulo 02 é desperdiçada). O lote de 32 é o
# meio-termo usado na prática: muitos passos por época e boa vetorização.

# %% [markdown]
# ### Solução do Exercício 3

# %%
rede_es = inicializar_rede(64, 64, 10, semente=0)
gerador = np.random.default_rng(0)
X_poucos, Y_poucos = X_treino[:100], Y_treino[:100]
melhor_perda = np.inf
melhor_rede, epoca_da_melhor = None, None

for epoca in range(300):
    ordem = gerador.permutation(len(X_poucos))
    for inicio in range(0, len(X_poucos), 32):
        lote = ordem[inicio:inicio + 32]
        P, memoria = propagacao_frente(X_poucos[lote], rede_es)
        gradientes = retropropagacao(Y_poucos[lote], rede_es, memoria)
        for nome in rede_es:
            rede_es[nome] = rede_es[nome] - 0.5 * gradientes[nome]
    perda_val, _ = avaliar(X_validacao, Y_validacao, rede_es)
    if perda_val < melhor_perda:
        melhor_perda = perda_val
        melhor_rede = {nome: valor.copy() for nome, valor in rede_es.items()}
        epoca_da_melhor = epoca

print(f"melhor rede: época {epoca_da_melhor}, acurácia de validação {avaliar(X_validacao, Y_validacao, melhor_rede)[1]:.1%}")
print(f"rede final:  época 299, acurácia de validação {avaliar(X_validacao, Y_validacao, rede_es)[1]:.1%}")

# %% [markdown]
# A rede guardada tem a menor perda de validação: ela é a mais "calibrada", sem a confiança exagerada da rede final.
# A acurácia das duas costuma ficar próxima, porque a acurácia só olha qual classe venceu, e não com que confiança.
# Na prática, a parada antecipada também economiza tempo: não é preciso treinar até o fim.
