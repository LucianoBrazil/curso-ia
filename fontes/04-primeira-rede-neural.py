# %% [markdown]
# # Módulo 04 — A primeira rede neural, do zero
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/04-primeira-rede-neural.ipynb)
#
# **Objetivo.** Entender o que é uma rede neural (neural network), como ela calcula uma resposta e como ela aprende,
# com a retropropagação (backpropagation). Tudo à mão, com NumPy.
#
# **O que você vai construir.** Uma rede com uma camada escondida que resolve um problema impossível para a regressão
# logística: o OU exclusivo (XOR). Você também vai conferir, número por número, que os gradientes calculados estão certos.

# %% [markdown]
# ## 1. A intuição: a lâmpada da escada
#
# Numa escada, a mesma lâmpada é controlada por dois interruptores: um embaixo, outro em cima. Você pode acender ou
# apagar a luz de qualquer um dos dois. O circuito (chamado de three-way) faz o seguinte:
#
# | interruptor de baixo | interruptor de cima | lâmpada |
# |---|---|---|
# | 0 | 0 | apagada (0) |
# | 0 | 1 | acesa (1) |
# | 1 | 0 | acesa (1) |
# | 1 | 1 | apagada (0) |
#
# A lâmpada acende quando os interruptores estão em posições **diferentes**. Na lógica, isso é o OU exclusivo
# (exclusive or, XOR).
#
# Vamos imaginar que um circuito lê a tensão de cada interruptor (perto de 0 V ou de 1 V), mas com ruído. Queremos um modelo
# que, a partir das duas tensões, diga se a lâmpada deve acender.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

cantos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
lampada_nos_cantos = np.array([0, 1, 1, 0])   # a tabela do XOR
n_por_canto = 50

X = np.vstack([canto + rng.normal(0, 0.15, (n_por_canto, 2)) for canto in cantos])
y = np.repeat(lampada_nos_cantos, n_por_canto).reshape(-1, 1).astype(float)
print("forma de X:", X.shape, "  forma de y:", y.shape)

# %% [markdown]
# `np.vstack` empilha tabelas uma embaixo da outra. `np.repeat` repete cada rótulo 50 vezes.
# Neste módulo, $y$ é uma coluna (forma `(200, 1)`), e não um vetor solto: isso facilita as contas com matrizes.

# %%
def desenhar_pontos(eixo, X, y):
    """Lâmpada acesa em azul, apagada em vermelho."""
    acesa = y[:, 0] == 1
    eixo.scatter(X[acesa, 0], X[acesa, 1], color="tab:blue", s=15, label="acesa (1)")
    eixo.scatter(X[~acesa, 0], X[~acesa, 1], color="tab:red", marker="x", s=15, label="apagada (0)")
    eixo.set_xlabel("tensão do interruptor de baixo (V)")
    eixo.set_ylabel("tensão do interruptor de cima (V)")


fig, eixo = plt.subplots(figsize=(5.5, 5))
desenhar_pontos(eixo, X, y)
eixo.legend()
eixo.grid(True)
plt.show()

# %% [markdown]
# (O `~` inverte uma máscara booleana: troca `True` por `False` e vice-versa.)
#
# Tente traçar, de cabeça, **uma única reta** que deixe os azuis de um lado e os vermelhos do outro. Não dá:
# os azuis estão em cantos opostos. Dizemos que os dados não são linearmente separáveis (linearly separable).

# %% [markdown]
# ## 2. A regressão logística falha
#
# A regressão logística do Módulo 03 só sabe desenhar fronteiras retas. Vamos confirmar. As funções abaixo foram
# copiadas do Módulo 03 (de forma compacta).

# %%
def sigmoide(z):
    """Vimos isso no Módulo 03."""
    return 1 / (1 + np.exp(-z))


def entropia_cruzada(p, y):
    """Vimos isso no Módulo 03: entropia cruzada média."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def treinar_logistica(X, y, taxa=0.5, passos=2000):
    """Vimos isso no Módulo 03: regressão logística por gradiente descendente."""
    w, b = np.zeros((X.shape[1], 1)), 0.0
    for passo in range(passos):
        residuo = sigmoide(X @ w + b) - y
        w = w - taxa * X.T @ residuo / len(y)
        b = b - taxa * np.mean(residuo)
    return w, b


w_log, b_log = treinar_logistica(X, y)
acuracia_logistica = np.mean((sigmoide(X @ w_log + b_log) >= 0.5) == y)
print(f"acurácia da regressão logística: {acuracia_logistica:.1%}")

# %% [markdown]
# Chutar sempre "apagada" já daria 50%. A regressão logística não faz muito melhor que isso. Vamos ver o motivo no mapa de
# probabilidades. A função de desenho recebe outra função como argumento (a que calcula a probabilidade). A palavra `lambda`
# cria uma função pequena numa linha só.

# %%
def desenhar_regioes(eixo, calcular_probabilidade, X, y, titulo):
    """Pinta a probabilidade prevista em cada ponto do plano e desenha a fronteira p = 0,5."""
    grade_1, grade_2 = np.meshgrid(np.linspace(-0.6, 1.6, 200), np.linspace(-0.6, 1.6, 200))
    pontos = np.column_stack([grade_1.ravel(), grade_2.ravel()])
    probabilidade = calcular_probabilidade(pontos).reshape(grade_1.shape)
    eixo.contourf(grade_1, grade_2, probabilidade, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.7)
    eixo.contour(grade_1, grade_2, probabilidade, levels=[0.5], colors="black", linewidths=2)
    desenhar_pontos(eixo, X, y)
    eixo.set_title(titulo)


fig, eixo = plt.subplots(figsize=(5.5, 5))
desenhar_regioes(eixo, lambda pontos: sigmoide(pontos @ w_log + b_log), X, y,
                 f"Regressão logística: {acuracia_logistica:.0%}")
plt.show()

# %% [markdown]
# Uma reta sempre deixa pelo menos um dos quatro grupos do lado errado. Precisamos de algo mais poderoso.

# %% [markdown]
# ## 3. O neurônio artificial
#
# Um neurônio artificial (artificial neuron) faz duas coisas:
#
# 1. calcula uma soma ponderada das entradas, mais um viés: $z = w_1 x_1 + w_2 x_2 + b$;
# 2. passa o resultado por uma função de ativação (activation function): $a = f(z)$.
#
# Com $f$ = sigmoide, um neurônio **é** a regressão logística do Módulo 03. A inspiração vem dos neurônios do cérebro,
# que recebem sinais, somam e "disparam" quando a soma passa de um limite. Mas a semelhança para por aí: um neurônio
# artificial é só essa conta.
#
# A ideia nova é **ligar vários neurônios**: as saídas de alguns viram as entradas de outros.

# %% [markdown]
# ## 4. Resolvendo o XOR à mão, com três neurônios
#
# A lógica ajuda: "XOR" é o mesmo que "pelo menos um ligado" **e não** "os dois ligados".
#
# - Neurônio $h_1$ (OU, OR): $z = 20 x_1 + 20 x_2 - 10$. Fica perto de 1 se pelo menos uma entrada é 1.
# - Neurônio $h_2$ (E, AND): $z = 20 x_1 + 20 x_2 - 30$. Fica perto de 1 só se as duas entradas são 1.
# - Neurônio de saída: $z = 20 h_1 - 20 h_2 - 10$. Fica perto de 1 se $h_1$ está ligado e $h_2$ desligado.
#
# **Exemplo à mão**, entrada $(1, 0)$, usando $\sigma(10) \approx 1$ e $\sigma(-10) \approx 0$:
#
# - $h_1 = \sigma(20 + 0 - 10) = \sigma(10) \approx 1$;
# - $h_2 = \sigma(20 + 0 - 30) = \sigma(-10) \approx 0$;
# - saída $= \sigma(20 \cdot 1 - 20 \cdot 0 - 10) = \sigma(10) \approx 1$. Lâmpada acesa. Correto.
#
# Os neurônios $h_1$ e $h_2$ formam uma camada escondida (hidden layer): não são nem a entrada nem a saída.
# A saída é a camada de saída (output layer).

# %%
W1_manual = np.array([[20.0, 20.0],
                      [20.0, 20.0]])      # coluna 0: neurônio OU; coluna 1: neurônio E
b1_manual = np.array([-10.0, -30.0])
W2_manual = np.array([[20.0], [-20.0]])
b2_manual = np.array([-10.0])

for entrada in cantos:
    h = sigmoide(entrada @ W1_manual + b1_manual)
    saida = sigmoide(h @ W2_manual + b2_manual)
    print(f"entrada {entrada}: h1 = {h[0]:.3f}, h2 = {h[1]:.3f}  ->  lâmpada = {saida[0]:.3f}")

# %% [markdown]
# **O que a camada escondida faz?** Ela troca as coordenadas. Em vez de olhar para $(x_1, x_2)$, a saída olha para $(h_1, h_2)$.
# No espaço novo, os pontos que eram impossíveis de separar ficam separáveis por uma reta. É a engenharia de atributos do
# Módulo 02, com uma diferença enorme: logo mais, a rede vai **aprender sozinha** esses atributos.

# %%
H_manual = sigmoide(X @ W1_manual + b1_manual)

fig, (eixo_original, eixo_escondido) = plt.subplots(1, 2, figsize=(11, 4.5))
desenhar_pontos(eixo_original, X, y)
eixo_original.set_title("Espaço original: nenhuma reta separa")
eixo_original.grid(True)

desenhar_pontos(eixo_escondido, H_manual, y)
eixo_escondido.plot([0, 1], [-0.5, 0.5], color="black", linewidth=2, label="h1 − h2 = 0,5")
eixo_escondido.set_xlabel("h1 (neurônio OU)")
eixo_escondido.set_ylabel("h2 (neurônio E)")
eixo_escondido.set_xlim(-0.1, 1.1)
eixo_escondido.set_ylim(-0.1, 1.1)
eixo_escondido.set_title("Espaço da camada escondida: uma reta separa")
eixo_escondido.legend()
eixo_escondido.grid(True)
plt.show()

# %% [markdown]
# ## 5. Camadas, em notação de matriz
#
# Com $n$ exemplos, 2 entradas e $k$ neurônios escondidos, a rede faz:
#
# $$ Z_1 = X W_1 + b_1 \qquad H = f(Z_1) \qquad Z_2 = H W_2 + b_2 \qquad P = \sigma(Z_2) $$
#
# | objeto | forma | o que é |
# |---|---|---|
# | $X$ | $n \times 2$ | entradas, uma linha por exemplo |
# | $W_1$ | $2 \times k$ | pesos da camada escondida, uma coluna por neurônio |
# | $b_1$ | $k$ | um viés por neurônio |
# | $H$ | $n \times k$ | as ativações escondidas (os "atributos aprendidos") |
# | $W_2$ | $k \times 1$ | pesos da saída |
# | $P$ | $n \times 1$ | probabilidade de a lâmpada acender |
#
# Calcular $P$ a partir de $X$, camada por camada, é a propagação para frente (forward pass).
#
# ### Por que a função de ativação é indispensável
#
# Sem $f$, teríamos $Z_2 = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2)$. Isso é só **outra** combinação
# linear de $X$: duas camadas lineares equivalem a uma. A ativação é a "dobra" que permite fronteiras curvas.
#
# ### Duas ativações
#
# - **Sigmoide:** $\sigma(z) = 1/(1 + e^{-z})$, derivada $\sigma(z)\,(1 - \sigma(z))$.
# - **ReLU** (rectified linear unit): $\text{ReLU}(z) = \max(0, z)$. Zero para negativos; o próprio $z$ para positivos.
#   Derivada: 0 se $z < 0$ e 1 se $z > 0$. É a ativação mais usada nas camadas escondidas hoje: simples e rápida.
#
# Exemplo: $\text{ReLU}(-3) = 0$, $\text{ReLU}(2{,}5) = 2{,}5$.

# %%
def relu(z):
    """Zero para negativos, o próprio valor para positivos."""
    return np.maximum(0, z)


def derivada_relu(z):
    """1 onde z > 0, 0 onde z <= 0."""
    return (z > 0).astype(float)


valores_z = np.linspace(-4, 4, 200)
fig, (eixo_f, eixo_df) = plt.subplots(1, 2, figsize=(11, 3.5))
eixo_f.plot(valores_z, sigmoide(valores_z), label="sigmoide")
eixo_f.plot(valores_z, relu(valores_z), label="ReLU")
eixo_f.set_title("Funções de ativação")
eixo_f.set_ylim(-0.5, 4)
eixo_f.legend()
eixo_f.grid(True)
eixo_df.plot(valores_z, sigmoide(valores_z) * (1 - sigmoide(valores_z)), label="derivada da sigmoide")
eixo_df.plot(valores_z, derivada_relu(valores_z), label="derivada da ReLU")
eixo_df.set_title("Suas derivadas")
eixo_df.legend()
eixo_df.grid(True)
plt.show()

# %% [markdown]
# Repare que a derivada da sigmoide nunca passa de 0,25. Isso vai importar daqui a pouco.

# %% [markdown]
# ## 6. Retropropagação: um exemplo à mão
#
# Para treinar, precisamos do gradiente da perda em relação a **cada** peso, inclusive os da camada escondida.
# A ferramenta é a regra da cadeia (chain rule), que apareceu no Módulo 03 como engrenagens: se $L$ depende de $a$, e $a$
# depende de $w$, então
#
# $$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w} $$
#
# Numa rede, a cadeia é longa: $w_1 \to z_1 \to h \to z_2 \to p \to L$. A retropropagação (backpropagation) calcula as
# inclinações **de trás para frente**: começa na perda e volta, elo por elo, reaproveitando o que já calculou.
#
# **A menor rede possível.** Uma entrada, um neurônio escondido (sigmoide), uma saída (sigmoide). Valores:
# $x = 1$, $w_1 = 0{,}5$, $b_1 = 0$, $w_2 = 1$, $b_2 = 0$, e a resposta certa é $y = 1$.
#
# **Para frente:**
#
# - $z_1 = w_1 x + b_1 = 0{,}5$, e $h = \sigma(0{,}5) \approx 0{,}6225$
# - $z_2 = w_2 h + b_2 = 0{,}6225$, e $p = \sigma(0{,}6225) \approx 0{,}6508$
# - $L = -\log p \approx 0{,}4296$
#
# **Para trás**, elo por elo:
#
# 1. $\frac{\partial L}{\partial z_2} = p - y = 0{,}6508 - 1 = -0{,}3492$ (o resultado simples do Módulo 03).
# 2. $\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z_2} \cdot h = -0{,}3492 \cdot 0{,}6225 \approx -0{,}2174$, e $\frac{\partial L}{\partial b_2} = -0{,}3492$.
# 3. $\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} \cdot w_2 = -0{,}3492 \cdot 1 = -0{,}3492$ (o erro "volta" para a camada escondida pelo peso $w_2$).
# 4. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot h\,(1 - h) = -0{,}3492 \cdot 0{,}6225 \cdot 0{,}3775 \approx -0{,}0821$ (passou pela derivada da sigmoide).
# 5. $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \cdot x = -0{,}0821$, e $\frac{\partial L}{\partial b_1} = -0{,}0821$.
#
# Todos os gradientes são negativos: aumentar qualquer peso aumentaria $p$ e diminuiria a perda. Faz sentido, pois queremos $p \to 1$.
# Repare também que o gradiente de $w_1$ é bem menor que o de $w_2$: ao voltar pela sigmoide, ele foi multiplicado por
# $h(1-h) \le 0{,}25$. Em redes muito profundas, isso se repete a cada camada e o gradiente pode quase sumir. Esse é um
# dos motivos para usar ReLU nas camadas escondidas.

# %%
x, y_mini = 1.0, 1.0
w1, b1, w2, b2 = 0.5, 0.0, 1.0, 0.0

# Para frente
z1 = w1 * x + b1
h = sigmoide(z1)
z2 = w2 * h + b2
p = sigmoide(z2)
perda = -np.log(p)
print(f"h = {h:.4f}   p = {p:.4f}   perda = {perda:.4f}")

# Para trás
d_z2 = p - y_mini
d_w2, d_b2 = d_z2 * h, d_z2
d_h = d_z2 * w2
d_z1 = d_h * h * (1 - h)
d_w1, d_b1 = d_z1 * x, d_z1
print(f"dL/dw2 = {d_w2:.4f}   dL/db2 = {d_b2:.4f}   dL/dw1 = {d_w1:.4f}   dL/db1 = {d_b1:.4f}")

# %% [markdown]
# **Conferindo com o passinho.** Mexemos um pouquinho em $w_1$ e vemos quanto a perda muda, como no Módulo 01.

# %%
def perda_mini(w1, b1, w2, b2):
    """Perda da rede mínima, para conferir os gradientes."""
    return -np.log(sigmoide(w2 * sigmoide(w1 * x + b1) + b2))


passinho = 1e-5
print(f"dL/dw1 pelo passinho: {(perda_mini(w1 + passinho, b1, w2, b2) - perda_mini(w1 - passinho, b1, w2, b2)) / (2 * passinho):.4f}")
print(f"dL/dw2 pelo passinho: {(perda_mini(w1, b1, w2 + passinho, b2) - perda_mini(w1, b1, w2 - passinho, b2)) / (2 * passinho):.4f}")

# %% [markdown]
# ## 7. A rede completa em código
#
# As mesmas cinco etapas, agora com matrizes, para $n$ exemplos e $k$ neurônios escondidos (ReLU).
# Cada linha da retropropagação é um elo da corrente da seção 6:
#
# | elo | exemplo à mão | com matrizes |
# |---|---|---|
# | perda → $Z_2$ | $p - y$ | $dZ_2 = (P - Y)/n$ |
# | $Z_2$ → $W_2$, $b_2$ | $\cdot\, h$ | $dW_2 = H^T dZ_2$, $\;db_2 = $ soma de $dZ_2$ |
# | $Z_2$ → $H$ | $\cdot\, w_2$ | $dH = dZ_2\, W_2^T$ |
# | $H$ → $Z_1$ | $\cdot\, f'(z_1)$ | $dZ_1 = dH \odot f'(Z_1)$ |
# | $Z_1$ → $W_1$, $b_1$ | $\cdot\, x$ | $dW_1 = X^T dZ_1$, $\;db_1 = $ soma de $dZ_1$ |
#
# O símbolo $\odot$ é a multiplicação posição por posição (o `*` do NumPy). A divisão por $n$ vem da média da perda.
#
# Os pesos começam **aleatórios e pequenos**. Se todos começassem iguais, todos os neurônios escondidos calculariam a mesma
# coisa, receberiam o mesmo gradiente e continuariam iguais para sempre: a rede teria, na prática, um neurônio só.
# Os parâmetros ficam guardados num dicionário (dictionary) do Python: uma coleção de valores com nome, como `rede["W1"]`.

# %%
def inicializar_rede(n_entradas, n_ocultos, semente=0):
    """Pesos aleatórios pequenos, vieses zerados."""
    gerador = np.random.default_rng(semente)
    return {
        "W1": gerador.normal(0, 1, (n_entradas, n_ocultos)),
        "b1": np.zeros(n_ocultos),
        "W2": gerador.normal(0, 1, (n_ocultos, 1)),
        "b2": np.zeros(1),
    }


def propagacao_frente(X, rede):
    """Calcula a probabilidade de saída e guarda os valores intermediários para a volta."""
    Z1 = X @ rede["W1"] + rede["b1"]
    H = relu(Z1)
    Z2 = H @ rede["W2"] + rede["b2"]
    P = sigmoide(Z2)
    memoria = {"X": X, "Z1": Z1, "H": H, "P": P}
    return P, memoria


def retropropagacao(y, rede, memoria):
    """Gradiente da entropia cruzada média em relação a cada parâmetro, de trás para frente."""
    n = len(y)
    dZ2 = (memoria["P"] - y) / n
    dW2 = memoria["H"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dH = dZ2 @ rede["W2"].T
    dZ1 = dH * derivada_relu(memoria["Z1"])
    dW1 = memoria["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


rede_teste = inicializar_rede(2, 8)
P_teste, memoria_teste = propagacao_frente(X, rede_teste)
gradientes_teste = retropropagacao(y, rede_teste, memoria_teste)
for nome, gradiente in gradientes_teste.items():
    print(f"gradiente de {nome}: forma {gradiente.shape}")

# %% [markdown]
# ## 8. Conferência numérica dos gradientes
#
# A retropropagação é fácil de errar: um índice trocado, uma transposta esquecida. Antes de treinar, fazemos a conferência
# numérica dos gradientes (gradient checking): para **cada** parâmetro, o passinho para cada lado, comparando com a fórmula.
# É lento (duas propagações por parâmetro), por isso só serve para conferir, nunca para treinar.
#
# `np.ndindex` percorre todas as posições de uma matriz, uma por uma.

# %%
def conferir_gradientes(X, y, rede, passinho=1e-5):
    """Compara os gradientes da retropropagação com os calculados pelo passinho para cada lado."""
    P, memoria = propagacao_frente(X, rede)
    analiticos = retropropagacao(y, rede, memoria)
    for nome in rede:
        numericos = np.zeros_like(rede[nome])
        for posicao in np.ndindex(rede[nome].shape):
            original = rede[nome][posicao]
            rede[nome][posicao] = original + passinho
            perda_mais = entropia_cruzada(propagacao_frente(X, rede)[0], y)
            rede[nome][posicao] = original - passinho
            perda_menos = entropia_cruzada(propagacao_frente(X, rede)[0], y)
            rede[nome][posicao] = original   # devolve o valor original
            numericos[posicao] = (perda_mais - perda_menos) / (2 * passinho)
        maior_diferenca = np.max(np.abs(numericos - analiticos[nome]))
        print(f"{nome}: maior diferença entre fórmula e passinho = {maior_diferenca:.2e}")


conferir_gradientes(X, y, inicializar_rede(2, 8))

# %% [markdown]
# Diferenças da ordem de $10^{-10}$ ou menores significam que a retropropagação está certa. Um erro no código daria
# diferenças grandes, de $10^{-2}$ ou mais.

# %% [markdown]
# ## 9. Treinando a rede no XOR
#
# O treino é o gradiente descendente de sempre, aplicado a todos os parâmetros: $\theta \leftarrow \theta - \eta \, \nabla_\theta L$,
# onde $\theta$ (letra grega "teta") representa qualquer parâmetro.

# %%
def treinar_rede(X, y, n_ocultos, taxa, passos, semente=0):
    """Gradiente descendente numa rede de uma camada escondida. Devolve a rede e o histórico da perda."""
    rede = inicializar_rede(X.shape[1], n_ocultos, semente)
    historico_perda = []
    for passo in range(passos):
        P, memoria = propagacao_frente(X, rede)
        historico_perda.append(entropia_cruzada(P, y))
        gradientes = retropropagacao(y, rede, memoria)
        for nome in rede:
            rede[nome] = rede[nome] - taxa * gradientes[nome]
    return rede, np.array(historico_perda)


def acuracia_rede(X, y, rede):
    """Fração de acertos, com limiar 0,5."""
    return np.mean((propagacao_frente(X, rede)[0] >= 0.5) == y)


rede, historico_perda = treinar_rede(X, y, n_ocultos=8, taxa=0.5, passos=3000)
print(f"perda final: {historico_perda[-1]:.4f}   acurácia: {acuracia_rede(X, y, rede):.1%}")
print(f"(regressão logística: {acuracia_logistica:.1%})")

# %% [markdown]
# **O aprendizado acontecendo.** À esquerda, a perda caindo. À direita, a fronteira de decisão em quatro momentos do treino.
# Como os pesos iniciais vêm da mesma semente, treinar de novo por menos passos mostra exatamente como a rede estava naquele momento.

# %%
plt.figure(figsize=(7, 3.5))
plt.plot(historico_perda)
plt.yscale("log")
plt.xlabel("passo")
plt.ylabel("entropia cruzada")
plt.title("A perda caindo")
plt.grid(True)
plt.show()

fig, eixos = plt.subplots(1, 4, figsize=(18, 4.5))
for eixo, passos in zip(eixos, [1, 30, 100, 3000]):
    rede_parcial, _ = treinar_rede(X, y, n_ocultos=8, taxa=0.5, passos=passos)
    desenhar_regioes(eixo, lambda pontos: propagacao_frente(pontos, rede_parcial)[0], X, y,
                     f"passo {passos}: acurácia {acuracia_rede(X, y, rede_parcial):.0%}")
plt.tight_layout()
plt.show()

# %% [markdown]
# No passo 1, com pesos aleatórios, a fronteira é uma forma sem sentido. Em algumas dezenas de passos ela se reorganiza
# numa faixa que isola os dois cantos azuis. Depois disso, o treino deixa a rede mais confiante: a faixa de transição entre
# azul e vermelho fica cada vez mais estreita. Cada neurônio ReLU contribui com uma "dobra" reta; juntas, as dobras formam
# uma fronteira que nenhuma regressão logística consegue desenhar.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Com a rede montada à mão da seção 4, calcule à mão $h_1$, $h_2$ e a saída para a entrada $(1, 1)$.
# Use $\sigma(30) \approx 1$, $\sigma(10) \approx 1$ e $\sigma(-10) \approx 0$. A lâmpada acende? Confira no código.
#
# **Exercício 2 (médio).** Treine redes com 1, 2, 3 e 8 neurônios escondidos (taxa 0,5, 3000 passos, semente 0) e mostre a
# acurácia de cada uma. Desenhe as regiões de decisão. Por que um único neurônio não basta?
#
# **Exercício 3 (desafio).** Troque a ReLU da camada escondida pela sigmoide. Você precisa mudar a propagação para frente
# **e** a retropropagação (a derivada da ativação muda). Rode `conferir_gradientes` adaptado para ter certeza de que acertou.
# Depois treine com 8 neurônios, taxa 0,5 e 3000 passos e compare a curva da perda com a da ReLU.

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | neural network | rede neural | *A neural network is a chain of simple neurons.* |
# | neuron | neurônio | *Each neuron computes a weighted sum.* |
# | layer | camada | *This network has two layers of weights.* |
# | hidden layer | camada escondida | *The hidden layer learns new features.* |
# | output layer | camada de saída | *The output layer gives the probability.* |
# | activation function | função de ativação | *Without an activation function, the network is linear.* |
# | ReLU | ReLU | *ReLU sets negative values to zero.* |
# | exclusive or (XOR) | ou exclusivo | *XOR is true when the inputs are different.* |
# | linearly separable | linearmente separável | *XOR is not linearly separable.* |
# | forward pass | propagação para frente | *The forward pass computes the prediction.* |
# | backpropagation | retropropagação | *Backpropagation sends the error backwards.* |
# | chain rule | regra da cadeia | *Backpropagation is the chain rule, applied many times.* |
# | gradient checking | conferência numérica dos gradientes | *Gradient checking catches bugs in backpropagation.* |
# | initialization | inicialização | *Random initialization breaks the symmetry.* |

# %% [markdown]
# ## Resumo
#
# - Um neurônio é uma soma ponderada seguida de uma função de ativação; com a sigmoide, é a regressão logística.
# - O XOR não é linearmente separável; uma camada escondida cria atributos novos em que ele passa a ser.
# - Sem ativação, várias camadas equivalem a uma só; a ativação (sigmoide, ReLU) é o que permite fronteiras curvas.
# - A retropropagação aplica a regra da cadeia de trás para frente, elo por elo, reaproveitando as contas.
# - A conferência numérica dos gradientes garante que a retropropagação está certa antes de treinar.
#
# **Próximo módulo:** a mesma rede, maior, vai aprender a reconhecer dígitos escritos à mão.

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
# - $h_1 = \sigma(20 + 20 - 10) = \sigma(30) \approx 1$
# - $h_2 = \sigma(20 + 20 - 30) = \sigma(10) \approx 1$
# - saída $= \sigma(20 \cdot 1 - 20 \cdot 1 - 10) = \sigma(-10) \approx 0$. A lâmpada fica apagada, como manda a tabela.

# %%
entrada_ex1 = np.array([1.0, 1.0])
h_ex1 = sigmoide(entrada_ex1 @ W1_manual + b1_manual)
print("h1, h2 =", np.round(h_ex1, 5), "  saída =", np.round(sigmoide(h_ex1 @ W2_manual + b2_manual), 5))

# %% [markdown]
# ### Solução do Exercício 2

# %%
fig, eixos = plt.subplots(1, 4, figsize=(18, 4.5))
for eixo, n_ocultos in zip(eixos, [1, 2, 3, 8]):
    rede_ex2, _ = treinar_rede(X, y, n_ocultos=n_ocultos, taxa=0.5, passos=3000)
    acc = acuracia_rede(X, y, rede_ex2)
    print(f"{n_ocultos} neurônio(s) escondido(s): acurácia {acc:.1%}")
    desenhar_regioes(eixo, lambda pontos: propagacao_frente(pontos, rede_ex2)[0], X, y, f"{n_ocultos} neurônio(s): {acc:.0%}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Com um neurônio, a camada escondida produz um único atributo, e a saída é uma função crescente (ou decrescente) dele:
# a fronteira continua sendo uma reta. Com 2 neurônios já é possível resolver o XOR (a rede manual tinha 2), mas o treino
# pode ficar preso numa solução ruim, dependendo dos pesos iniciais. Com mais neurônios, há mais "dobras" disponíveis
# e o treino fica mais confiável.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def propagacao_frente_sigmoide(X, rede):
    """Igual à propagação_frente, mas com sigmoide na camada escondida."""
    Z1 = X @ rede["W1"] + rede["b1"]
    H = sigmoide(Z1)
    P = sigmoide(H @ rede["W2"] + rede["b2"])
    return P, {"X": X, "Z1": Z1, "H": H, "P": P}


def retropropagacao_sigmoide(y, rede, memoria):
    """Igual à retropropagacao, mas a derivada da ativação escondida é H (1 - H)."""
    n = len(y)
    dZ2 = (memoria["P"] - y) / n
    dH = dZ2 @ rede["W2"].T
    dZ1 = dH * memoria["H"] * (1 - memoria["H"])
    return {"W1": memoria["X"].T @ dZ1, "b1": dZ1.sum(axis=0),
            "W2": memoria["H"].T @ dZ2, "b2": dZ2.sum(axis=0)}


# Conferência numérica, só para W1 (a parte que mudou)
rede_s = inicializar_rede(2, 8)
analitico = retropropagacao_sigmoide(y, rede_s, propagacao_frente_sigmoide(X, rede_s)[1])["W1"]
numerico = np.zeros_like(rede_s["W1"])
for posicao in np.ndindex(rede_s["W1"].shape):
    original = rede_s["W1"][posicao]
    rede_s["W1"][posicao] = original + 1e-5
    perda_mais = entropia_cruzada(propagacao_frente_sigmoide(X, rede_s)[0], y)
    rede_s["W1"][posicao] = original - 1e-5
    perda_menos = entropia_cruzada(propagacao_frente_sigmoide(X, rede_s)[0], y)
    rede_s["W1"][posicao] = original
    numerico[posicao] = (perda_mais - perda_menos) / 2e-5
print(f"W1: maior diferença = {np.max(np.abs(numerico - analitico)):.2e}")

# Treino com sigmoide
rede_s = inicializar_rede(2, 8)
historico_sigmoide = []
for passo in range(3000):
    P, memoria = propagacao_frente_sigmoide(X, rede_s)
    historico_sigmoide.append(entropia_cruzada(P, y))
    gradientes = retropropagacao_sigmoide(y, rede_s, memoria)
    for nome in rede_s:
        rede_s[nome] = rede_s[nome] - 0.5 * gradientes[nome]

print(f"perda final: ReLU = {historico_perda[-1]:.4f}   sigmoide = {historico_sigmoide[-1]:.4f}")
plt.figure(figsize=(7, 3.5))
plt.plot(historico_perda, label="ReLU")
plt.plot(historico_sigmoide, label="sigmoide")
plt.yscale("log")
plt.xlabel("passo")
plt.ylabel("entropia cruzada")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Com a sigmoide na camada escondida, a perda cai mais devagar. Parte do motivo é o que vimos na seção 6: ao voltar
# pela sigmoide, o gradiente é multiplicado por no máximo 0,25.
