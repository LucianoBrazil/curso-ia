# %% [markdown]
# # Módulo 07 — PyTorch e a derivada automática
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/07-pytorch.ipynb)
#
# **Objetivo.** Conhecer o PyTorch, a biblioteca usada para criar boa parte das redes neurais modernas.
# A mensagem central: **ele faz exatamente o que você já fez à mão**, só que de forma automática e em escala.
#
# **O que você vai construir.** Você vai conferir que os gradientes do PyTorch batem com os da retropropagação feita à mão
# (Módulo 04), comparar otimizadores e refazer a rede de dígitos do Módulo 05 em poucas linhas.

# %% [markdown]
# ## 1. A intuição: uma calculadora de derivadas
#
# Lembre do trabalho que deu a retropropagação no Módulo 04: deduzir cada elo da regra da cadeia, escrever o código, conferir
# com o passinho. Se mudarmos a rede (outra ativação, mais uma camada), tudo precisa ser refeito.
#
# Um framework (framework) de aprendizado profundo, como o PyTorch, resolve isso. Você escreve só a **ida** (a propagação para
# frente). Ele anota cada operação e, quando você pede, calcula sozinho todos os gradientes da **volta**. Isso se chama
# derivação automática (automatic differentiation); no PyTorch, o mecanismo tem o apelido de autograd.
#
# O PyTorch já vem instalado no Colab. (Para rodar no seu computador: `pip install torch`.)

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

torch.manual_seed(0)   # semente do PyTorch, para os resultados se repetirem
print("versão do PyTorch:", torch.__version__)

# %% [markdown]
# ## 2. Tensores: os arrays do PyTorch
#
# O tensor (tensor) é o array do PyTorch. Quase tudo que você fazia com arrays do NumPy funciona igual.

# %%
a = torch.tensor([1.0, 2.0, 3.0])
M = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

print("a * 2      =", a * 2)
print("a @ a      =", a @ a)                       # produto escalar, como no Módulo 02
print("M @ [1, 1] =", M @ torch.tensor([1.0, 1.0]))
print("forma de M:", M.shape, "  tipo:", M.dtype)

# %% [markdown]
# Duas diferenças práticas em relação ao NumPy:
#
# - **Tipo dos números.** O PyTorch costuma usar `float32` (números reais com 32 bits), mais leve e suficiente para redes neurais.
#   O NumPy usa `float64` por padrão. Ao converter, escolhemos o tipo.
# - **Conversão.** `torch.tensor(array, dtype=torch.float32)` leva do NumPy para o PyTorch; `.numpy()` faz o caminho de volta.

# %%
array_numpy = np.array([1.5, 2.5])
tensor = torch.tensor(array_numpy, dtype=torch.float32)
print(tensor, "->", tensor.numpy())

# %% [markdown]
# ## 3. O autograd: derivadas sem fazer conta
#
# Para o PyTorch acompanhar um número, criamos o tensor com `requires_grad=True` ("precisa de gradiente").
# Depois calculamos a perda normalmente e chamamos `.backward()`. O gradiente aparece em `.grad`.
#
# **Exemplo à mão.** Pela fórmula do Módulo 01, a derivada de $L = (w\,t - d)^2$ em relação a $w$ é $2\,(w\,t - d)\,t$.
# Com $t = 3$, $d = 5$ e $w = 1$:
#
# $$ \frac{dL}{dw} = 2\,(3 - 5) \cdot 3 = 2 \cdot (-2) \cdot 3 = -12 $$

# %%
w = torch.tensor(1.0, requires_grad=True)
perda = (w * 3 - 5) ** 2
perda.backward()
print("perda:", perda.item(), "  dL/dw pelo PyTorch:", w.grad.item())

# %% [markdown]
# (`.item()` transforma um tensor de um único número num número comum do Python.)
#
# **Como ele faz isso?** Enquanto a conta é feita, o PyTorch monta um grafo computacional (computational graph): um registro de
# cada operação e de quem depende de quem. Aqui: $w \to w \cdot 3 \to (\cdot) - 5 \to (\cdot)^2 \to L$. O `.backward()` percorre
# esse registro de trás para frente, multiplicando as derivadas de cada elo. É a retropropagação do Módulo 04, feita pelo computador.

# %% [markdown]
# ## 4. A prova: o exemplo à mão do Módulo 04
#
# No Módulo 04 fizemos a retropropagação à mão numa rede mínima: uma entrada, um neurônio escondido (sigmoide) e uma saída
# (sigmoide), com $x = 1$, $w_1 = 0{,}5$, $b_1 = 0$, $w_2 = 1$, $b_2 = 0$ e resposta certa $y = 1$. Os resultados foram:
#
# | gradiente | à mão (Módulo 04) |
# |---|---|
# | $\partial L / \partial w_2$ | −0,2174 |
# | $\partial L / \partial b_2$ | −0,3492 |
# | $\partial L / \partial w_1$ | −0,0821 |
# | $\partial L / \partial b_1$ | −0,0821 |
#
# Agora escrevemos **só a ida** e deixamos o PyTorch fazer a volta.

# %%
x, y = torch.tensor(1.0), torch.tensor(1.0)
w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
b2 = torch.tensor(0.0, requires_grad=True)

h = torch.sigmoid(w1 * x + b1)
p = torch.sigmoid(w2 * h + b2)
perda = -torch.log(p)
perda.backward()

for nome, parametro in [("w2", w2), ("b2", b2), ("w1", w1), ("b1", b1)]:
    print(f"dL/d{nome} = {parametro.grad.item():.4f}")

# %% [markdown]
# Os quatro números batem com a conta feita à mão. O PyTorch não faz mágica: faz a regra da cadeia, sem cansar e sem errar.
#
# ### Uma armadilha: os gradientes se acumulam
#
# Cada `.backward()` **soma** o novo gradiente ao que já estava em `.grad`. Isso é útil em alguns casos avançados, mas no treino
# comum precisamos zerar o gradiente antes de cada passo. Veja o que acontece sem zerar:

# %%
w = torch.tensor(1.0, requires_grad=True)
for vez in range(3):
    perda = (w * 3 - 5) ** 2
    perda.backward()
    print(f"depois do backward número {vez + 1}: w.grad = {w.grad.item()}")

# %% [markdown]
# −12, −24, −36: os gradientes foram se somando. Por isso, todo laço de treino em PyTorch tem um passo de "zerar os gradientes".

# %% [markdown]
# ## 5. Gradiente descendente com autograd
#
# Vamos refazer o segundo experimento do Módulo 01 (o carro a 20 m/s que já estava a 30 m da marca), com dados parecidos.
# O laço tem três partes, as mesmas de sempre:
#
# 1. **ida:** calcular a perda;
# 2. **volta:** `perda.backward()` preenche `w.grad` e `b.grad`;
# 3. **passo:** $w \leftarrow w - \eta \cdot$ `w.grad` (e o mesmo para $b$), depois zerar os gradientes.
#
# O passo fica dentro de `torch.no_grad()`: é um aviso para o PyTorch **não** anotar essa conta no grafo, porque atualizar os pesos
# não faz parte do modelo.

# %%
rng = np.random.default_rng(42)
tempo_np = np.linspace(0, 10, 25)
distancia_np = 20 * tempo_np + 30 + rng.normal(0, 10, tempo_np.shape)
tempo = torch.tensor(tempo_np, dtype=torch.float32)
distancia = torch.tensor(distancia_np, dtype=torch.float32)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
taxa = 0.02
for passo in range(1000):
    perda = torch.mean((w * tempo + b - distancia) ** 2)   # ida
    perda.backward()                                      # volta
    with torch.no_grad():                                 # passo
        w -= taxa * w.grad
        b -= taxa * b.grad
    w.grad.zero_()
    b.grad.zero_()

w_exato, b_exato = np.polyfit(tempo_np, distancia_np, 1)
print(f"PyTorch:            w = {w.item():.3f}   b = {b.item():.3f}")
print(f"mínimos quadrados:  w = {w_exato:.3f}   b = {b_exato:.3f}")

# %% [markdown]
# Mesmo resultado da solução exata, e não escrevemos nenhuma derivada.

# %% [markdown]
# ## 6. Otimizadores: SGD, momento e Adam
#
# O passo "$\theta \leftarrow \theta - \eta \, \nabla L$" também vem pronto: é um otimizador (optimizer). O laço vira
#
# ```
# otimizador.zero_grad()   # zera os gradientes
# perda.backward()         # calcula os gradientes
# otimizador.step()        # dá o passo
# ```
#
# Além do gradiente descendente simples (`torch.optim.SGD`), há variações mais espertas. Duas ideias importantes:
#
# **Momento (momentum).** Pense numa bola rolando no vale do Módulo 01. Ela não para a cada passo: tem inércia. Em vez de andar
# só pela inclinação atual, o otimizador anda por uma "velocidade" $m$ que acumula as inclinações recentes:
#
# $$ m \leftarrow \beta\, m + \nabla L \qquad\qquad \theta \leftarrow \theta - \eta\, m $$
#
# com $\beta$ perto de 1 (por exemplo 0,9). Num vale comprido, as inclinações apontam sempre para o mesmo lado ao longo do vale:
# a velocidade cresce e a bola desce mais rápido. Nas paredes, as inclinações trocam de sinal e tendem a se cancelar
# (mas a inércia também pode fazer a bola balançar de uma parede à outra, como você vai ver).
#
# **Exemplo à mão**, com $\beta = 0{,}9$ e o gradiente valendo 1 em três passos seguidos: $m = 1$, depois $0{,}9 + 1 = 1{,}9$,
# depois $1{,}71 + 1 = 2{,}71$. O passo vai crescendo enquanto a inclinação insiste na mesma direção.
#
# **Adam.** Combina o momento com um tamanho de passo **próprio para cada parâmetro**: divide o passo pelo tamanho típico dos
# gradientes recentes daquele parâmetro. Parâmetros com gradientes enormes andam com cuidado; parâmetros com gradientes pequenos
# andam mais. É como se o Adam fizesse, sozinho, um pouco da normalização do Módulo 02. Na versão simplificada:
#
# $$ m \leftarrow \beta_1 m + (1 - \beta_1)\, \nabla L \qquad v \leftarrow \beta_2 v + (1 - \beta_2)\, (\nabla L)^2 \qquad \theta \leftarrow \theta - \eta \, \frac{m}{\sqrt{v} + \varepsilon} $$
#
# ($\varepsilon$ é um número minúsculo que evita divisão por zero; a versão completa tem ainda uma pequena correção nos primeiros passos.)
# O Adam é o otimizador mais usado hoje, e funciona bem com a taxa padrão de 0,001 em muitos problemas.
#
# Vamos comparar os três no vale estreito do Módulo 01, **sem** normalizar os dados.

# %%
def treinar_reta(criar_otimizador, passos=200):
    """Ajusta d = w t + b com o otimizador dado. Devolve o caminho (w, b) e as perdas."""
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    otimizador = criar_otimizador([w, b])
    caminho, perdas = [], []
    for passo in range(passos):
        perda = torch.mean((w * tempo + b - distancia) ** 2)
        caminho.append((w.item(), b.item()))
        perdas.append(perda.item())
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    return np.array(caminho), np.array(perdas)


otimizadores = {
    "SGD (taxa 0,02)": lambda parametros: torch.optim.SGD(parametros, lr=0.02),
    "SGD com momento 0,9 (taxa 0,02)": lambda parametros: torch.optim.SGD(parametros, lr=0.02, momentum=0.9),
    "Adam (taxa 1)": lambda parametros: torch.optim.Adam(parametros, lr=1.0),
}
resultados = {nome: treinar_reta(criar) for nome, criar in otimizadores.items()}
for nome, (caminho, perdas) in resultados.items():
    print(f"{nome:32s} após 200 passos: w = {caminho[-1, 0]:6.2f}   b = {caminho[-1, 1]:6.2f}   perda = {perdas[-1]:8.2f}")

# %% [markdown]
# **O aprendizado acontecendo**, para os três otimizadores: o caminho no mapa do erro (como no Módulo 01) e a perda caindo.

# %%
grade_w, grade_b = np.meshgrid(np.linspace(-2, 40, 150), np.linspace(-20, 70, 150))
mapa_erro = np.mean((grade_w[..., None] * tempo_np + grade_b[..., None] - distancia_np) ** 2, axis=-1)
niveis = np.logspace(np.log10(mapa_erro.min()), np.log10(mapa_erro.max()), 25)

fig, (eixo_mapa, eixo_perda) = plt.subplots(1, 2, figsize=(13, 4.5))
eixo_mapa.contour(grade_w, grade_b, mapa_erro, levels=niveis, cmap="viridis")
for nome, (caminho, perdas) in resultados.items():
    eixo_mapa.plot(caminho[:, 0], caminho[:, 1], marker=".", markersize=3, label=nome)
    eixo_perda.plot(perdas, label=nome)
eixo_mapa.scatter([w_exato], [b_exato], color="black", zorder=3, label="mínimo")
eixo_mapa.set_xlabel("w (m/s)")
eixo_mapa.set_ylabel("b (m)")
eixo_mapa.set_title("Caminhos no mapa do erro")
eixo_mapa.legend(fontsize=8, loc="lower right")
eixo_perda.set_yscale("log")
eixo_perda.set_xlabel("passo")
eixo_perda.set_ylabel("erro quadrático médio")
eixo_perda.set_title("A perda caindo")
eixo_perda.legend(fontsize=8)
eixo_perda.grid(True)
plt.show()

# %% [markdown]
# (Para o mapa usamos um truque do NumPy: `[..., None]` acrescenta um eixo, e a conta é feita para toda a grade de uma vez.)
#
# O SGD simples desce rápido até o fundo do vale e depois se arrasta ao longo dele, como no Módulo 01. Com momento, a bola ganha
# inércia: balança de uma parede à outra do vale, mas avança ao longo dele e chega ao mínimo nos 200 passos. O Adam, que ajusta o
# passo de cada parâmetro, segue um caminho bem mais direto, sem o zigue-zague, e só desacelera perto do fim. Os dois vencem o SGD
# simples neste vale. Nenhum otimizador é o melhor em todos os problemas; o Adam é o ponto de partida mais comum.

# %% [markdown]
# ## 7. Montando redes com peças prontas
#
# O PyTorch traz as camadas prontas, no módulo `torch.nn`:
#
# - `nn.Linear(entradas, saidas)`: a conta $XW + b$, com os pesos já criados e inicializados;
# - `nn.ReLU()`: a ativação ReLU;
# - `nn.Sequential(...)`: liga as peças em sequência, a saída de uma vira a entrada da próxima.
#
# A rede do Módulo 05 (64 entradas, 64 neurônios escondidos, 10 saídas) fica assim:

# %%
def criar_rede_digitos():
    """A rede do Módulo 05: 64 -> 64 (ReLU) -> 10."""
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


rede = criar_rede_digitos()
print(rede)
for nome, parametro in rede.named_parameters():
    print(f"{nome:10s} forma {tuple(parametro.shape)}")

# %% [markdown]
# Um detalhe: o PyTorch guarda o peso de `nn.Linear` "deitado", com forma (saídas, entradas), o contrário do nosso $W_1$. É só
# uma convenção; por dentro, a conta é a mesma.
#
# **A perda.** `nn.CrossEntropyLoss` recebe as **pontuações** brutas da última camada (chamadas logits) e o rótulo como número
# inteiro (não precisa de one-hot). Ela aplica a softmax por dentro, de um jeito numericamente estável, e calcula a entropia
# cruzada do Módulo 05. Por isso a rede não tem softmax no fim.

# %% [markdown]
# ### O PyTorch confere com a nossa retropropagação
#
# Antes de treinar, a prova mais forte: pegamos os pesos dessa rede, calculamos os gradientes com a retropropagação em NumPy do
# Módulo 05 e comparamos com os do PyTorch, nas mesmas 20 imagens. Usamos `float64` nos dois lados, para a comparação ser exata.

# %%
from sklearn.datasets import load_digits

digitos = load_digits()
X_todos = digitos.data / 16.0
rotulos = digitos.target


def softmax_np(Z):
    """Vimos isso no Módulo 05."""
    exponenciais = np.exp(Z - Z.max(axis=1, keepdims=True))
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


def gradientes_numpy(X, Y, W1, b1, W2, b2):
    """Propagação para frente e retropropagação do Módulo 05, em NumPy."""
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    P = softmax_np(H @ W2 + b2)
    dZ2 = (P - Y) / len(Y)
    dH = dZ2 @ W2.T
    dZ1 = dH * (Z1 > 0)
    return {"W1": X.T @ dZ1, "b1": dZ1.sum(axis=0), "W2": H.T @ dZ2, "b2": dZ2.sum(axis=0)}


rede_teste = criar_rede_digitos().double()       # float64
X_20, rotulos_20 = X_todos[:20], rotulos[:20]

# Gradientes pelo PyTorch
perda = nn.CrossEntropyLoss()(rede_teste(torch.tensor(X_20)), torch.tensor(rotulos_20))
perda.backward()

# Gradientes pela nossa retropropagação, com os mesmos pesos (transpostos para a nossa convenção)
camada1, camada2 = rede_teste[0], rede_teste[2]
nossos = gradientes_numpy(X_20, np.eye(10)[rotulos_20],
                          camada1.weight.detach().numpy().T, camada1.bias.detach().numpy(),
                          camada2.weight.detach().numpy().T, camada2.bias.detach().numpy())

print("W1:", np.max(np.abs(nossos["W1"] - camada1.weight.grad.numpy().T)))
print("b1:", np.max(np.abs(nossos["b1"] - camada1.bias.grad.numpy())))
print("W2:", np.max(np.abs(nossos["W2"] - camada2.weight.grad.numpy().T)))
print("b2:", np.max(np.abs(nossos["b2"] - camada2.bias.grad.numpy())))

# %% [markdown]
# Diferenças da ordem de $10^{-17}$: os mesmos números, até o limite da precisão do computador.
# (`np.eye(10)[rotulos]` é um jeito curto de fazer o one-hot; `.detach()` tira um tensor do grafo antes de convertê-lo para NumPy.)

# %% [markdown]
# ## 8. A rede de dígitos do Módulo 05, em PyTorch
#
# Mesma divisão dos dados do Módulo 05 (mesma semente), mesma rede, mesmos minilotes de 32. Compare o laço de treino abaixo
# com as dezenas de linhas das funções `propagacao_frente` e `retropropagacao` do Módulo 05.

# %%
gerador_divisao = np.random.default_rng(42)
ordem = gerador_divisao.permutation(len(rotulos))
indices_treino, indices_validacao, indices_teste = ordem[:1078], ordem[1078:1437], ordem[1437:]


def para_tensores(indices):
    """Imagens (float32) e rótulos (inteiros) de um pedaço dos dados."""
    return torch.tensor(X_todos[indices], dtype=torch.float32), torch.tensor(rotulos[indices])


X_treino, y_treino = para_tensores(indices_treino)
X_validacao, y_validacao = para_tensores(indices_validacao)
X_teste, y_teste = para_tensores(indices_teste)
print(f"treino: {len(X_treino)}   validação: {len(X_validacao)}   teste: {len(X_teste)}")


def acuracia(rede, X, y):
    """Fração de acertos; sem anotar nada no grafo."""
    with torch.no_grad():
        return (rede(X).argmax(dim=1) == y).float().mean().item()


def treinar_digitos(rede, otimizador, epocas=40, tamanho_lote=32, semente=0):
    """Treino com minilotes. Guarda a perda de treino e a acurácia de validação de cada época."""
    funcao_perda = nn.CrossEntropyLoss()
    gerador = torch.Generator().manual_seed(semente)
    historico = {"perda_treino": [], "acc_val": []}
    for epoca in range(epocas):
        ordem_lotes = torch.randperm(len(X_treino), generator=gerador)
        for inicio in range(0, len(X_treino), tamanho_lote):
            lote = ordem_lotes[inicio:inicio + tamanho_lote]
            perda = funcao_perda(rede(X_treino[lote]), y_treino[lote])   # ida
            otimizador.zero_grad()
            perda.backward()                                            # volta
            otimizador.step()                                           # passo
        with torch.no_grad():
            historico["perda_treino"].append(funcao_perda(rede(X_treino), y_treino).item())
        historico["acc_val"].append(acuracia(rede, X_validacao, y_validacao))
    return historico

# %% [markdown]
# Treinamos duas vezes, com a mesma inicialização: com SGD (taxa 0,1, como no Módulo 05) e com Adam (taxa 0,001, a padrão).

# %%
redes, historicos = {}, {}
for nome in ["SGD", "Adam"]:
    torch.manual_seed(0)                       # mesma inicialização para os dois
    redes[nome] = criar_rede_digitos()
    if nome == "SGD":
        otimizador = torch.optim.SGD(redes[nome].parameters(), lr=0.1)
    else:
        otimizador = torch.optim.Adam(redes[nome].parameters(), lr=0.001)
    inicio = time.perf_counter()
    historicos[nome] = treinar_digitos(redes[nome], otimizador)
    print(f"{nome:4s}: {time.perf_counter() - inicio:.1f} s   acurácia de validação final {historicos[nome]['acc_val'][-1]:.1%}")

# %%
fig, (eixo_perda, eixo_acc) = plt.subplots(1, 2, figsize=(12, 4))
for nome, historico in historicos.items():
    eixo_perda.plot(historico["perda_treino"], label=nome)
    eixo_acc.plot(historico["acc_val"], label=nome)
eixo_perda.set_yscale("log")
eixo_perda.set_xlabel("época")
eixo_perda.set_ylabel("entropia cruzada (treino)")
eixo_perda.legend()
eixo_perda.grid(True)
eixo_acc.set_xlabel("época")
eixo_acc.set_ylabel("acurácia de validação")
eixo_acc.legend()
eixo_acc.grid(True)
fig.suptitle("Rede de dígitos em PyTorch: SGD contra Adam")
plt.show()

# %% [markdown]
# **A prova final.** Escolhemos a rede pela **validação** (Módulo 05) e só então olhamos o teste, uma única vez.

# %%
escolhida = max(historicos, key=lambda nome: historicos[nome]["acc_val"][-1])
print(f"rede escolhida pela validação: {escolhida}")
print(f"acurácia no teste: {acuracia(redes[escolhida], X_teste, y_teste):.1%}   (Módulo 05, em NumPy: 95,6%)")

# %% [markdown]
# ## 9. NumPy à mão contra PyTorch
#
# | etapa | Módulos 04 e 05 (NumPy) | PyTorch |
# |---|---|---|
# | criar a rede | `inicializar_rede`, pesos sorteados à mão | `nn.Sequential(nn.Linear, nn.ReLU, ...)` |
# | ida | `propagacao_frente`: $Z_1$, ReLU, $Z_2$, softmax | `rede(X)` |
# | perda | `entropia_cruzada` com one-hot | `nn.CrossEntropyLoss()` |
# | volta | `retropropagacao`: um elo da regra da cadeia por linha | `perda.backward()` |
# | conferir a volta | `conferir_gradientes` | desnecessário (mas você conferiu na seção 7) |
# | passo | laço `rede[nome] -= taxa * gradiente` | `otimizador.step()` |
#
# Nada de novo acontece por dentro. A diferença é que, para mudar a rede, basta mudar a ida: a volta se ajusta sozinha.
# É isso que permite treinar redes com centenas de camadas e bilhões de parâmetros, como os modelos de linguagem dos próximos módulos.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Para $L = (w\,t - d)^2$ com $t = 2$, $d = 5$ e $w = 3$, calcule à mão $dL/dw$ com a fórmula do Módulo 01.
# Confira com o autograd e com o passinho para cada lado.
#
# **Exercício 2 (médio).** Treine a rede de dígitos por 20 épocas com SGD (taxas 0,01; 0,1; 0,5) e com Adam (taxas 0,0001; 0,001; 0,01),
# sempre com `torch.manual_seed(0)` antes de criar a rede. Mostre a acurácia de validação final de cada combinação.
# Qual otimizador depende mais da escolha da taxa?
#
# **Exercício 3 (desafio).** Acrescente uma segunda camada escondida: 64 → 128 → 64 → 10, com ReLU entre as camadas. (À mão, isso
# exigiria reescrever a retropropagação; aqui, é mudar a ida.) Conte os parâmetros das duas redes com
# `sum(p.numel() for p in rede.parameters())`, treine a nova com Adam (taxa 0,001, 40 épocas) e compare a acurácia de **validação**
# com a da rede de uma camada.

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | framework | framework / biblioteca | *PyTorch is a deep learning framework.* |
# | tensor | tensor | *A tensor is like a NumPy array.* |
# | automatic differentiation | derivação automática | *Automatic differentiation computes gradients for us.* |
# | autograd | autograd | *Autograd records every operation.* |
# | computational graph | grafo computacional | *The computational graph is walked backwards.* |
# | requires grad | precisa de gradiente | *Set requires_grad to track a tensor.* |
# | backward | volta (retropropagação) | *Calling backward fills the grad attribute.* |
# | optimizer | otimizador | *The optimizer updates the weights.* |
# | momentum | momento | *Momentum works like inertia.* |
# | Adam | Adam | *Adam adapts the step size of each parameter.* |
# | logits | logits (pontuações) | *The loss function receives the logits.* |
# | layer | camada | *A linear layer computes XW plus b.* |
# | parameter | parâmetro | *This network has about five thousand parameters.* |
# | zero the gradients | zerar os gradientes | *Always zero the gradients before backward.* |

# %% [markdown]
# ## Resumo
#
# - Tensores são os arrays do PyTorch; quase tudo funciona como no NumPy, e a conversão é simples.
# - Com `requires_grad=True` e `.backward()`, o autograd calcula os gradientes pela regra da cadeia, percorrendo o grafo computacional.
# - Os gradientes do PyTorch batem com a retropropagação à mão (Módulo 04) e com a do NumPy (Módulo 05).
# - Otimizadores fazem o passo; momento dá inércia e Adam adapta o passo de cada parâmetro, vencendo o vale estreito do Módulo 01.
# - O framework faz o que você fez à mão, só que em escala: mudar a rede é mudar só a ida.
#
# **Próximo módulo:** linguagem. Vamos transformar texto em números e escrever em inglês com um modelo de bigramas.

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
# $\frac{dL}{dw} = 2\,(w\,t - d)\,t = 2\,(6 - 5) \cdot 2 = 4$.

# %%
w = torch.tensor(3.0, requires_grad=True)
perda = (w * 2 - 5) ** 2
perda.backward()


def perda_ex1(w):
    """A mesma perda, como função comum do Python."""
    return (w * 2 - 5) ** 2


passinho = 1e-4
print(f"autograd: {w.grad.item():.4f}   passinho: {(perda_ex1(3 + passinho) - perda_ex1(3 - passinho)) / (2 * passinho):.4f}")

# %% [markdown]
# ### Solução do Exercício 2

# %%
combinacoes = [("SGD", 0.01), ("SGD", 0.1), ("SGD", 0.5), ("Adam", 0.0001), ("Adam", 0.001), ("Adam", 0.01)]
for nome, taxa in combinacoes:
    torch.manual_seed(0)
    rede_ex2 = criar_rede_digitos()
    classe = torch.optim.SGD if nome == "SGD" else torch.optim.Adam
    historico_ex2 = treinar_digitos(rede_ex2, classe(rede_ex2.parameters(), lr=taxa), epocas=20)
    print(f"{nome:4s} taxa {taxa:<7}: acurácia de validação {historico_ex2['acc_val'][-1]:.1%}")

# %% [markdown]
# Neste problema pequeno, os dois têm sensibilidade parecida: com a taxa pequena demais, aprendem pouco em 20 épocas; com as
# maiores, vão bem. A vantagem do Adam aparece em redes maiores, com gradientes de tamanhos muito diferentes em cada camada:
# como ele divide o passo pelo tamanho típico dos gradientes de cada parâmetro, a mesma taxa serve para todos. Na prática,
# quase sempre se começa pelo Adam com 0,001 e só depois se ajusta. E repare: a escolha foi feita pela **validação**.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def criar_rede_profunda():
    """Duas camadas escondidas: 64 -> 128 -> 64 -> 10."""
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def contar_parametros(rede):
    """Número total de pesos e vieses."""
    return sum(p.numel() for p in rede.parameters())


torch.manual_seed(0)
rede_profunda = criar_rede_profunda()
historico_profunda = treinar_digitos(rede_profunda, torch.optim.Adam(rede_profunda.parameters(), lr=0.001))
print(f"uma camada:   {contar_parametros(redes['Adam']):6d} parâmetros   validação {historicos['Adam']['acc_val'][-1]:.1%}")
print(f"duas camadas: {contar_parametros(rede_profunda):6d} parâmetros   validação {historico_profunda['acc_val'][-1]:.1%}")

# %% [markdown]
# A rede de duas camadas tem mais de três vezes mais parâmetros e, aqui, não ganha nada (fica até um pouco abaixo na validação):
# 1078 imagens de 8×8 já são bem resolvidas por uma camada, e mais parâmetros também significam mais chance de sobreajuste. Redes mais profundas brilham com dados maiores e mais
# complexos. O ponto principal do exercício é outro: a mudança foi só na ida, e o PyTorch cuidou da volta.
