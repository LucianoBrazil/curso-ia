# %% [markdown]
# # Módulo 02 — Várias variáveis: vetores, matrizes e normalização
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/02-varias-variaveis.ipynb)
#
# **Objetivo.** Passar de uma variável para várias ao mesmo tempo, usando vetores e matrizes.
# Você vai ver por que normalizar os dados acelera muito o gradiente descendente.
#
# **O que você vai construir.** Um modelo que descobre a potência dos aparelhos de uma casa só olhando a conta de luz
# de cada dia. Depois, um modelo "linear" que ajusta uma parábola e mede a gravidade.

# %% [markdown]
# ## 1. A intuição: a conta de luz
#
# Da Física: energia é potência vezes tempo, $E = P \cdot t$. Um chuveiro de 5,4 kW ligado por 1 hora gasta 5,4 kWh.
#
# O consumo de um dia é a **soma** do que cada aparelho gastou, mais um consumo fixo (geladeira, aparelhos em espera):
#
# $$ \text{consumo} = w_1 \cdot (\text{minutos de banho}) + w_2 \cdot (\text{horas de ar-condicionado}) + w_3 \cdot (\text{lavagens de roupa}) + b $$
#
# Cada peso $w_j$ diz quanto aquele aparelho gasta por unidade de uso. O viés $b$ é o consumo fixo.
#
# **Exemplo à mão.** Pesos $w = (0{,}09;\ 1{,}0;\ 0{,}8)$ e $b = 2{,}5$. Num dia com 20 minutos de banho,
# 3 horas de ar-condicionado e 1 lavagem:
#
# $$ 0{,}09 \cdot 20 + 1{,}0 \cdot 3 + 0{,}8 \cdot 1 + 2{,}5 = 1{,}8 + 3 + 0{,}8 + 2{,}5 = 8{,}1 \text{ kWh} $$
#
# O problema, de novo, é o inverso: temos o consumo de muitos dias e queremos descobrir os pesos.
# Cada coisa que medimos (minutos de banho, horas de ar...) é um atributo (feature). O que queremos prever (o consumo)
# é o alvo (target). Um modelo assim, com vários atributos, é uma regressão linear múltipla (multiple linear regression).

# %% [markdown]
# ## 2. Vetores e o produto escalar
#
# Um vetor (vector) é uma lista ordenada de números. Na Física, um vetor 2D como $(3, 4)$ é uma seta.
# Aqui, um vetor é só uma "ficha" com vários números: os atributos de um dia, $x = (20, 3, 1)$, ou os pesos, $w = (0{,}09;\ 1{,}0;\ 0{,}8)$.
#
# A conta da seção 1 tem um nome: produto escalar (dot product). Multiplique posição por posição e some tudo:
#
# $$ w \cdot x = \sum_{j=1}^{m} w_j\, x_j = w_1 x_1 + w_2 x_2 + \dots + w_m x_m $$
#
# onde $m$ é o número de atributos. No NumPy, o produto escalar é `np.dot(w, x)` ou, mais curto, `w @ x`.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt

w_exemplo = np.array([0.09, 1.0, 0.8])
x_dia = np.array([20, 3, 1])
b_exemplo = 2.5

print("posição por posição:", w_exemplo * x_dia)
print("somando tudo:       ", np.sum(w_exemplo * x_dia))
print("com @:              ", w_exemplo @ x_dia)
print("consumo previsto:   ", w_exemplo @ x_dia + b_exemplo, "kWh")

# %% [markdown]
# ## 3. Matrizes: a tabela de dados
#
# Uma matriz (matrix) é uma tabela de números. Vamos guardar os dados assim: **cada linha é um dia (um exemplo),
# cada coluna é um atributo**. Com $n$ dias e $m$ atributos, a matriz $X$ tem $n$ linhas e $m$ colunas.
#
# Multiplicar a matriz pelo vetor de pesos, $Xw$, faz o produto escalar de **cada linha** com $w$, de uma só vez:
#
# $$ \begin{pmatrix} 20 & 3 & 1 \\ 40 & 0 & 2 \end{pmatrix} \begin{pmatrix} 0{,}09 \\ 1{,}0 \\ 0{,}8 \end{pmatrix}
# = \begin{pmatrix} 0{,}09 \cdot 20 + 1{,}0 \cdot 3 + 0{,}8 \cdot 1 \\ 0{,}09 \cdot 40 + 1{,}0 \cdot 0 + 0{,}8 \cdot 2 \end{pmatrix}
# = \begin{pmatrix} 5{,}6 \\ 5{,}2 \end{pmatrix} $$
#
# Somando o viés $b = 2{,}5$ em cada linha, as previsões são 8,1 e 7,7 kWh. O modelo inteiro, para todos os dias, cabe numa linha:
#
# $$ \hat{y} = X w + b $$
#
# O chapéu em $\hat{y}$ (lê-se "y chapéu") indica previsão. O $y$ sem chapéu é o valor medido.

# %%
X_exemplo = np.array([[20, 3, 1],
                      [40, 0, 2]])
print("forma da matriz:", X_exemplo.shape)   # (linhas, colunas)
print("previsões:", X_exemplo @ w_exemplo + b_exemplo)

# %% [markdown]
# ## 4. Os dados: 200 dias de uma casa
#
# Vamos simular 200 dias. Os valores reais dos pesos ficam guardados, mas o modelo **não** vai olhar para eles:
#
# - chuveiro: 0,09 kWh por minuto (5,4 kW);
# - ar-condicionado: 1,0 kWh por hora (1 kW);
# - máquina de lavar: 0,8 kWh por lavagem;
# - consumo fixo: 2,5 kWh por dia.
#
# O medidor tem ruído de cerca de 0,5 kWh. A função `np.column_stack` junta vetores lado a lado, como colunas de uma tabela.

# %%
rng = np.random.default_rng(42)
n_dias = 200

minutos_banho = rng.uniform(5, 60, n_dias)
horas_ar = rng.uniform(0, 10, n_dias)
lavagens = rng.integers(0, 3, n_dias)          # 0, 1 ou 2 lavagens

pesos_reais = np.array([0.09, 1.0, 0.8])       # o computador não olha
vies_real = 2.5

X = np.column_stack([minutos_banho, horas_ar, lavagens])
y = X @ pesos_reais + vies_real + rng.normal(0, 0.5, n_dias)

nomes_atributos = ["minutos de banho", "horas de ar-condicionado", "lavagens de roupa"]
print("forma de X:", X.shape, "  forma de y:", y.shape)
print("primeiros 3 dias:")
print(X[:3])
print(y[:3])

# %% [markdown]
# Um gráfico do consumo contra cada atributo. Nenhum atributo, sozinho, explica o consumo: os pontos ficam espalhados
# porque os outros aparelhos também variam. O modelo precisa olhar para os três ao mesmo tempo.

# %%
fig, eixos = plt.subplots(1, 3, figsize=(15, 4))
for j, eixo in enumerate(eixos):
    eixo.scatter(X[:, j], y, s=12, alpha=0.7)
    eixo.set_xlabel(nomes_atributos[j])
    eixo.set_ylabel("consumo do dia (kWh)")
    eixo.grid(True)
plt.show()

# %% [markdown]
# `X[:, j]` significa "todas as linhas, coluna $j$": é a coluna inteira de um atributo.

# %% [markdown]
# ## 5. O modelo e o erro
#
# O modelo é $\hat{y} = Xw + b$ e o erro é o mesmo erro quadrático médio do Módulo 01:
#
# $$ E(w, b) = \frac{1}{n} \sum_{i=1}^{n} \big( \hat{y}_i - y_i \big)^2 $$
#
# A diferença é que agora $w$ é um vetor com 3 números. A paisagem do erro tem 4 dimensões ($w_1, w_2, w_3, b$),
# impossível de desenhar. Mas o gradiente descendente não precisa do desenho: só precisa da inclinação.

# %%
def prever(X, w, b):
    """Regressão linear múltipla: uma previsão por linha de X."""
    return X @ w + b


def erro_quadratico_medio(previsto, medido):
    """Média dos quadrados das diferenças (vimos isso nos Módulos 00 e 01)."""
    return np.mean((previsto - medido) ** 2)


print(f"erro com todos os pesos zerados: {erro_quadratico_medio(prever(X, np.zeros(3), 0.0), y):.2f} kWh²")

# %% [markdown]
# ## 6. O gradiente como vetor
#
# No Módulo 01, com $\hat{y} = w\,t + b$, vimos que $\frac{\partial E}{\partial w} = \frac{2}{n}\sum_i e_i\, t_i$, com o resíduo $e_i = \hat{y}_i - y_i$.
# Com vários pesos, a conta é a mesma para cada um: o atributo $t_i$ é trocado pela coluna $j$, $x_{ij}$.
#
# $$ \frac{\partial E}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} e_i\, x_{ij} \qquad\qquad \frac{\partial E}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i $$
#
# O gradiente (gradient) é o vetor com todas essas derivadas: $\nabla_w E = \left( \frac{\partial E}{\partial w_1}, \frac{\partial E}{\partial w_2}, \frac{\partial E}{\partial w_3} \right)$.
# Ele aponta para onde o erro sobe mais depressa; o gradiente descendente anda no sentido oposto.
#
# **A transposta.** Repare que $\sum_i e_i\, x_{ij}$ é o produto escalar da coluna $j$ com o vetor de resíduos. Fazer isso
# para todas as colunas de uma vez é virar a tabela de lado (colunas viram linhas) e multiplicar pelo vetor $e$.
# A tabela virada é a transposta (transpose), $X^T$, no NumPy `X.T`:
#
# $$ \nabla_w E = \frac{2}{n}\, X^T e $$
#
# **Exemplo à mão**, com os 2 dias da seção 3. Previsões 8,1 e 7,7; suponha medidos 8,5 e 7,2. Resíduos: $e = (-0{,}4;\ +0{,}5)$.
#
# $$ \frac{\partial E}{\partial w_1} = \frac{2}{2}\big( (-0{,}4)(20) + (0{,}5)(40) \big) = -8 + 20 = 12 $$
#
# $$ \frac{\partial E}{\partial w_2} = (-0{,}4)(3) + (0{,}5)(0) = -1{,}2 \qquad \frac{\partial E}{\partial w_3} = (-0{,}4)(1) + (0{,}5)(2) = 0{,}6 \qquad \frac{\partial E}{\partial b} = -0{,}4 + 0{,}5 = 0{,}1 $$
#
# Guarde isto: a derivada do peso dos **minutos** (12) é muito maior que as outras, só porque minutos são números grandes.
# Isso vai causar um problema na seção 7.

# %%
def gradientes(X, y, w, b):
    """Gradiente do erro quadrático médio em relação a w (vetor) e a b (número)."""
    n = len(y)
    residuo = prever(X, w, b) - y
    gradiente_w = (2 / n) * (X.T @ residuo)
    gradiente_b = (2 / n) * np.sum(residuo)
    return gradiente_w, gradiente_b


# Conferindo o exemplo feito à mão
y_exemplo = np.array([8.5, 7.2])
gradiente_w_exemplo, gradiente_b_exemplo = gradientes(X_exemplo, y_exemplo, w_exemplo, b_exemplo)
print("gradiente de w:", np.round(gradiente_w_exemplo, 4), "  gradiente de b:", round(gradiente_b_exemplo, 4))

# %% [markdown]
# ### Vetorização: a mesma conta, com laços e sem laços
#
# Para ver que `X.T @ residuo` faz mesmo o que a fórmula diz, vamos calcular o gradiente com dois laços `for`,
# termo a termo, e comparar. Depois medimos o tempo das duas versões numa tabela grande.

# %%
def gradiente_w_com_lacos(X, y, w, b):
    """Mesma conta do gradiente de w, termo a termo, sem vetorização."""
    n, m = X.shape
    residuo = prever(X, w, b) - y
    gradiente_w = np.zeros(m)
    for j in range(m):
        soma = 0.0
        for i in range(n):
            soma += residuo[i] * X[i, j]
        gradiente_w[j] = 2 / n * soma
    return gradiente_w


w_teste = np.array([0.05, 0.5, 0.5])
print("com laços:  ", gradiente_w_com_lacos(X, y, w_teste, 1.0))
print("vetorizado: ", gradientes(X, y, w_teste, 1.0)[0])

X_grande = np.tile(X, (500, 1))   # repete a tabela 500 vezes: 100 mil linhas
y_grande = np.tile(y, 500)

inicio = time.perf_counter()
gradiente_w_com_lacos(X_grande, y_grande, w_teste, 1.0)
tempo_lacos = time.perf_counter() - inicio

inicio = time.perf_counter()
gradientes(X_grande, y_grande, w_teste, 1.0)
tempo_vetorizado = time.perf_counter() - inicio

print(f"laços: {tempo_lacos:.3f} s   vetorizado: {tempo_vetorizado:.5f} s")

# %% [markdown]
# Mesmos números, tempos muito diferentes. Uma rede neural faz essa conta milhões de vezes: sem vetorização, seria inviável.

# %% [markdown]
# ## 7. Treinando sem normalizar
#
# O gradiente descendente é o mesmo do Módulo 01, só que agora $w$ é um vetor e todos os pesos andam juntos:
#
# $$ w \leftarrow w - \eta\, \nabla_w E \qquad\qquad b \leftarrow b - \eta\, \frac{\partial E}{\partial b} $$

# %%
def treinar_regressao(X, y, taxa, passos):
    """Gradiente descendente para regressão linear múltipla. Devolve w, b e o histórico do erro."""
    w = np.zeros(X.shape[1])
    b = 0.0
    historico_erro = [erro_quadratico_medio(prever(X, w, b), y)]
    for passo in range(passos):
        gradiente_w, gradiente_b = gradientes(X, y, w, b)
        w = w - taxa * gradiente_w
        b = b - taxa * gradiente_b
        historico_erro.append(erro_quadratico_medio(prever(X, w, b), y))
    return w, b, np.array(historico_erro)


w_bruto, b_bruto, erro_bruto = treinar_regressao(X, y, taxa=0.0005, passos=300)
print("pesos aprendidos:", np.round(w_bruto, 3), " viés:", round(b_bruto, 3))
print("pesos reais:     ", pesos_reais, " viés:", vies_real)

# %% [markdown]
# Depois de 300 passos, os pesos ainda estão longe dos reais, principalmente o da lavagem e o consumo fixo.
# Por que não aumentar a taxa? Porque, com uma taxa um pouco maior, o método **diverge** (o Exercício 2 mostra isso).
#
# O culpado é a escala (scale). Os minutos de banho vão até 60; as lavagens, só até 2. O gradiente na direção dos minutos
# é enorme (lembre do 12 contra 0,6 na seção 6). A taxa precisa ser pequena para não "explodir" nessa direção,
# e aí os outros pesos andam devagar demais. É o vale comprido e estreito do Módulo 01, agora com mais dimensões.

# %% [markdown]
# ## 8. Normalização: colocando todos na mesma escala
#
# A solução é transformar cada atributo para que todos tenham escala parecida. A forma mais comum é a padronização
# (standardization), também chamada de normalização pelo escore z (z-score):
#
# $$ x' = \frac{x - \mu}{\sigma} $$
#
# - $\mu$ (letra grega "mi") é a média (mean) da coluna.
# - $\sigma$ (letra grega "sigma") é o desvio padrão (standard deviation): o tamanho típico da distância até a média,
#   $\sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}$.
#
# Depois disso, toda coluna tem média 0 e desvio padrão 1: os valores ficam, em geral, entre −2 e +2.
#
# **Exemplo à mão.** Minutos de banho de 3 dias: 20, 40, 30. Média $\mu = 30$.
#
# $$ \sigma = \sqrt{\frac{(20-30)^2 + (40-30)^2 + (30-30)^2}{3}} = \sqrt{\frac{200}{3}} \approx 8{,}16 $$
#
# $$ x' = \left( \frac{20-30}{8{,}16};\ \frac{40-30}{8{,}16};\ \frac{30-30}{8{,}16} \right) \approx (-1{,}22;\ +1{,}22;\ 0) $$
#
# No NumPy, `X.mean(axis=0)` calcula a média de cada coluna (`axis=0` significa "descendo pelas linhas").

# %%
def normalizar(X):
    """Padroniza cada coluna: média 0 e desvio padrão 1. Devolve também média e desvio, para desfazer depois."""
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    return (X - media) / desvio, media, desvio


print("exemplo à mão:", np.round(normalizar(np.array([[20.0], [40.0], [30.0]]))[0].ravel(), 2))

X_norm, media_X, desvio_X = normalizar(X)
print("médias das colunas:         ", np.round(X_norm.mean(axis=0), 3))
print("desvios padrão das colunas: ", np.round(X_norm.std(axis=0), 3))

# %% [markdown]
# Agora treinamos com os dados normalizados. Como as escalas estão equilibradas, podemos usar uma taxa 200 vezes maior.

# %%
w_norm, b_norm, erro_norm = treinar_regressao(X_norm, y, taxa=0.1, passos=300)
print(f"erro final sem normalizar: {erro_bruto[-1]:.3f} kWh²")
print(f"erro final normalizado:    {erro_norm[-1]:.3f} kWh²")

# %% [markdown]
# **O aprendizado acontecendo**, com e sem normalização, nos mesmos 300 passos. O eixo vertical está em escala logarítmica.

# %%
plt.figure(figsize=(8, 4))
plt.plot(erro_bruto, label="sem normalizar (taxa 0,0005)")
plt.plot(erro_norm, label="normalizado (taxa 0,1)")
plt.yscale("log")
plt.xlabel("passo")
plt.ylabel("erro quadrático médio (kWh²)")
plt.title("A normalização acelera o aprendizado")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# A curva normalizada despenca e chega ao piso (o ruído do medidor, cerca de $0{,}5^2 = 0{,}25$ kWh²).
# A outra desce devagar e termina num patamar bem mais alto.
#
# ### Voltando às unidades originais
#
# Os pesos aprendidos valem para os dados normalizados. Para ler a potência dos aparelhos, desfazemos a transformação:
#
# $$ w'\,\frac{x - \mu}{\sigma} = \frac{w'}{\sigma}\, x - \frac{w'\,\mu}{\sigma} \quad\Rightarrow\quad w = \frac{w'}{\sigma}, \qquad b = b' - \sum_j \frac{w'_j\, \mu_j}{\sigma_j} $$

# %%
def desnormalizar_pesos(w_norm, b_norm, media, desvio):
    """Converte pesos treinados em dados normalizados para as unidades originais."""
    w = w_norm / desvio
    b = b_norm - np.sum(w_norm * media / desvio)
    return w, b


w_final, b_final = desnormalizar_pesos(w_norm, b_norm, media_X, desvio_X)
for nome, aprendido, real in zip(nomes_atributos, w_final, pesos_reais):
    print(f"{nome:26s} aprendido = {aprendido:.3f}   real = {real:.3f}")
print(f"{'consumo fixo (viés)':26s} aprendido = {b_final:.3f}   real = {vies_real:.3f}")
print(f"\npotência do chuveiro: {w_final[0] * 60:.2f} kW (real: {pesos_reais[0] * 60:.2f} kW)")

# %% [markdown]
# O modelo descobriu a potência de cada aparelho sem ninguém contar a ele.
#
# **Conferência com a solução exata.** Como no Módulo 01, existe uma fórmula de mínimos quadrados (least squares) para
# várias variáveis. O NumPy a resolve com `np.linalg.lstsq`. Para incluir o viés, acrescentamos uma coluna de uns em $X$:
# o viés vira o peso dessa coluna.

# %%
X_com_uns = np.column_stack([X, np.ones(n_dias)])
solucao_exata = np.linalg.lstsq(X_com_uns, y, rcond=None)[0]
print("mínimos quadrados:     ", np.round(solucao_exata, 3))
print("gradiente descendente: ", np.round(np.append(w_final, b_final), 3))

# %% [markdown]
# ## 9. Engenharia de atributos: uma parábola com um modelo linear
#
# Um modelo linear só sabe fazer somas de atributos vezes pesos. Isso quer dizer que ele só ajusta retas? **Não.**
# Ele é linear nos **pesos**, mas os atributos podem ser qualquer coisa que nós calculemos a partir dos dados.
# Criar atributos novos e úteis é a engenharia de atributos (feature engineering).
#
# Exemplo de Física: uma bola lançada para cima a 15 m/s, de 1,5 m de altura. A altura é
#
# $$ h(t) = h_0 + v_0\, t - \tfrac{1}{2}\, g\, t^2 $$
#
# Se usarmos só o atributo $t$, o modelo é uma reta, e uma reta não descreve a subida e a descida.
# Mas se dermos ao modelo **duas** colunas, $t$ e $t^2$, ele pode aprender $h = w_1 t + w_2 t^2 + b$.
# Comparando com a Física: $w_1 = v_0$, $w_2 = -\frac{1}{2} g$ e $b = h_0$. O peso $w_2$ nos dá a gravidade: $g = -2\, w_2$.

# %%
tempo_bola = np.linspace(0, 3, 60)
altura_bola = 1.5 + 15 * tempo_bola - 0.5 * 9.8 * tempo_bola**2 + rng.normal(0, 0.4, tempo_bola.shape)

# Modelo 1: só o tempo (uma coluna). reshape(-1, 1) transforma o vetor numa tabela de uma coluna.
X_reta = tempo_bola.reshape(-1, 1)
# Modelo 2: tempo e tempo ao quadrado (duas colunas)
X_parabola = np.column_stack([tempo_bola, tempo_bola**2])

resultados = {}
for nome, X_modelo in [("reta (t)", X_reta), ("parábola (t e t²)", X_parabola)]:
    X_modelo_norm, media_m, desvio_m = normalizar(X_modelo)
    w_m, b_m, hist_m = treinar_regressao(X_modelo_norm, altura_bola, taxa=0.3, passos=1000)
    w_m, b_m = desnormalizar_pesos(w_m, b_m, media_m, desvio_m)
    resultados[nome] = (X_modelo, w_m, b_m)
    print(f"{nome:18s} erro final = {hist_m[-1]:6.2f} m²   pesos = {np.round(w_m, 3)}   viés = {b_m:.3f}")

w_parabola = resultados["parábola (t e t²)"][1]
print(f"\ngravidade estimada: g = {-2 * w_parabola[1]:.2f} m/s²   velocidade inicial: {w_parabola[0]:.2f} m/s")

# %%
plt.figure(figsize=(7, 4))
plt.scatter(tempo_bola, altura_bola, color="tab:red", label="medições")
for nome, (X_modelo, w_m, b_m) in resultados.items():
    plt.plot(tempo_bola, prever(X_modelo, w_m, b_m), label=f"modelo: {nome}")
plt.xlabel("tempo (s)")
plt.ylabel("altura (m)")
plt.title("Mesmo método, atributos diferentes")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Com o atributo $t^2$, o mesmo método de regressão linear acompanha a parábola e devolve $g$ perto de 9,8 m/s²
# (o ruído das medições impede de acertar em cheio).
# A ideia vai voltar com força: as redes neurais (Módulo 04) são, no fundo, máquinas que **aprendem sozinhas** os atributos certos.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Com $w = (0{,}09;\ 1{,}0;\ 0{,}8)$ e $b = 2{,}5$, calcule à mão o consumo previsto para dois dias:
# (10 min de banho, 8 h de ar, 0 lavagens) e (30 min, 2 h, 1 lavagem). Depois confira montando uma matriz $2 \times 3$ e usando `@`.
#
# **Exercício 2 (médio).** Treine **sem normalizar** (`treinar_regressao(X, y, ...)`), por 300 passos, com as taxas
# 0,0003; 0,0005; 0,0007; 0,0008 e 0,001. Mostre o erro final de cada uma. Qual é a maior taxa que não diverge?
# Algum desses erros chega perto do erro obtido com os dados normalizados?
#
# **Exercício 3 (desafio).** Um pêndulo simples de comprimento $L$ tem período $T = 2\pi\sqrt{L/g}$. Isso não é linear em $L$,
# mas elevando ao quadrado: $T^2 = \frac{4\pi^2}{g}\, L$. Gere 30 pêndulos com $L$ entre 0,2 e 2 m
# (`np.random.default_rng(7).uniform`), calcule $T$ com $g = 9{,}8$ e some um ruído de 0,02 s.
# Use engenharia de atributos (o alvo passa a ser $T^2$) e `treinar_regressao` com normalização para estimar $g$
# a partir do peso aprendido: $g = 4\pi^2 / w$.

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | vector | vetor | *A vector is an ordered list of numbers.* |
# | matrix | matriz | *Each row of the matrix is one day.* |
# | dot product | produto escalar | *The dot product multiplies and then adds.* |
# | transpose | transposta | *The transpose turns columns into rows.* |
# | feature | atributo | *Shower minutes is a feature.* |
# | target | alvo | *The target is the daily energy use.* |
# | multiple linear regression | regressão linear múltipla | *Multiple linear regression uses many features.* |
# | gradient vector | vetor gradiente | *The gradient vector points uphill.* |
# | vectorization | vetorização | *Vectorization made the code much faster.* |
# | scale | escala | *The features have very different scales.* |
# | standardization | padronização | *After standardization, each column has mean zero.* |
# | mean | média | *The mean of the column is thirty.* |
# | standard deviation | desvio padrão | *The standard deviation measures the spread.* |
# | feature engineering | engenharia de atributos | *Feature engineering turned a line into a parabola.* |

# %% [markdown]
# ## Resumo
#
# - Uma tabela de dados é uma matriz $X$: cada linha é um exemplo, cada coluna é um atributo.
# - O modelo linear com vários atributos é $\hat{y} = Xw + b$: um produto escalar por linha.
# - O gradiente é um vetor, $\nabla_w E = \frac{2}{n} X^T e$, e a vetorização faz essa conta muito mais rápido que laços.
# - Atributos em escalas diferentes deixam o gradiente descendente lento; padronizar ($x' = (x-\mu)/\sigma$) resolve.
# - Engenharia de atributos (como acrescentar $t^2$) deixa um modelo linear ajustar curvas.
#
# **Próximo módulo:** em vez de prever um número, vamos prever "sim ou não" com a regressão logística.

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
# Dia 1: $0{,}09 \cdot 10 + 1{,}0 \cdot 8 + 0{,}8 \cdot 0 + 2{,}5 = 0{,}9 + 8 + 0 + 2{,}5 = 11{,}4$ kWh.
#
# Dia 2: $0{,}09 \cdot 30 + 1{,}0 \cdot 2 + 0{,}8 \cdot 1 + 2{,}5 = 2{,}7 + 2 + 0{,}8 + 2{,}5 = 8{,}0$ kWh.

# %%
dias_ex1 = np.array([[10, 8, 0],
                     [30, 2, 1]])
print(dias_ex1 @ w_exemplo + b_exemplo)

# %% [markdown]
# ### Solução do Exercício 2

# %%
for taxa in [0.0003, 0.0005, 0.0007, 0.0008, 0.001]:
    _, _, hist = treinar_regressao(X, y, taxa=taxa, passos=300)
    print(f"taxa {taxa:.4f}: erro final = {hist[-1]:.4g}")
print(f"normalizado (taxa 0,1): erro final = {erro_norm[-1]:.4g}")

# %% [markdown]
# Até 0,0007 o método converge; a partir de 0,0008 ele diverge e o erro explode. Mesmo na melhor taxa possível,
# o erro sem normalizar fica muito acima do erro com normalização. (O `_` é um nome para valores que vamos ignorar.)

# %% [markdown]
# ### Solução do Exercício 3

# %%
rng7 = np.random.default_rng(7)
comprimento = rng7.uniform(0.2, 2.0, 30)
periodo = 2 * np.pi * np.sqrt(comprimento / 9.8) + rng7.normal(0, 0.02, 30)

X_pendulo = comprimento.reshape(-1, 1)   # atributo: L
alvo = periodo**2                        # alvo transformado: T²

X_pendulo_norm, media_p, desvio_p = normalizar(X_pendulo)
w_p, b_p, _ = treinar_regressao(X_pendulo_norm, alvo, taxa=0.3, passos=500)
w_p, b_p = desnormalizar_pesos(w_p, b_p, media_p, desvio_p)

print(f"peso aprendido: {w_p[0]:.4f} s²/m   viés: {b_p:.4f} s² (deveria ser perto de 0)")
print(f"g estimada: {4 * np.pi**2 / w_p[0]:.2f} m/s²")
