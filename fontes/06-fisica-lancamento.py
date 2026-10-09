# %% [markdown]
# # Módulo 06 — Física + IA: o lançamento oblíquo
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/06-fisica-lancamento.ipynb)
#
# **Objetivo.** Usar a rede neural dos módulos anteriores num problema de Física sem fórmula simples: o alcance de uma bola
# com resistência do ar. Você vai ver onde a IA brilha (rapidez) e onde ela falha (fora dos dados que viu).
#
# **O que você vai construir.** Um simulador pelo método de Euler, um conjunto de dados gerado por ele e uma rede, feita com NumPy,
# que aprende a prever o alcance de um chute a partir da velocidade e do ângulo.

# %% [markdown]
# ## 1. A intuição: o chute de longa distância
#
# Um jogador chuta a bola do chão, com velocidade $v_0$ e ângulo $\theta$ acima da horizontal. Até onde ela vai?
#
# No vácuo, a resposta tem fórmula. Na vida real, o ar freia a bola, e a fórmula deixa de valer. Não existe uma expressão
# simples para o alcance com resistência do ar. Mas podemos **simular** o voo no computador, passo a passo.
#
# E aí entra a pergunta deste módulo: se já temos um simulador, para que uma rede neural? Guarde a pergunta; a resposta
# aparece na seção 9.

# %% [markdown]
# ## 2. Sem resistência do ar: a fórmula do alcance
#
# Sem ar, a única força é o peso. Separando o movimento em horizontal e vertical (como no Módulo 00):
#
# $$ x(t) = v_0 \cos\theta \; t \qquad\qquad y(t) = v_0 \sin\theta \; t - \tfrac{1}{2} g\, t^2 $$
#
# A bola volta ao chão quando $y = 0$, ou seja, em $t = \frac{2 v_0 \sin\theta}{g}$. Nesse instante,
#
# $$ R = v_0 \cos\theta \cdot \frac{2 v_0 \sin\theta}{g} = \frac{v_0^2 \cdot 2 \sin\theta \cos\theta}{g} = \frac{v_0^2 \sin(2\theta)}{g} $$
#
# usando a identidade $2 \sin\theta \cos\theta = \sin(2\theta)$. O alcance é máximo quando $\sin(2\theta) = 1$, isto é, $\theta = 45°$.
#
# **Exemplo à mão.** $v_0 = 20$ m/s e $\theta = 45°$: $R = 400 \cdot 1 / 9{,}8 \approx 40{,}8$ m.
# Com $\theta = 30°$: $\sin 60° \approx 0{,}866$, então $R \approx 400 \cdot 0{,}866 / 9{,}8 \approx 35{,}3$ m.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt

g = 9.8   # m/s²


def alcance_sem_ar(v0, angulo_graus):
    """Alcance (m) no vácuo: v0² sen(2θ) / g."""
    return v0**2 * np.sin(2 * np.radians(angulo_graus)) / g


print(f"v0 = 20 m/s, 45°: {alcance_sem_ar(20, 45):.1f} m")
print(f"v0 = 20 m/s, 30°: {alcance_sem_ar(20, 30):.1f} m")

# %% [markdown]
# (`np.radians` converte graus para radianos, a unidade que as funções trigonométricas do NumPy usam.)

# %% [markdown]
# ## 3. Com resistência do ar: não há fórmula simples
#
# Em velocidades altas, a força de arrasto (drag) do ar é proporcional ao **quadrado** da velocidade e aponta contra o movimento.
# Dividindo pela massa, a aceleração causada pelo ar é $-c\,|\vec{v}|\,\vec{v}$, onde $|\vec{v}| = \sqrt{v_x^2 + v_y^2}$ é o
# módulo da velocidade. Somando a gravidade:
#
# $$ a_x = -c\,|\vec{v}|\, v_x \qquad\qquad a_y = -g - c\,|\vec{v}|\, v_y $$
#
# A constante $c$ depende da densidade do ar, do tamanho, da forma e da massa da bola. Para uma bola de futebol,
# $c \approx 0{,}013$ m⁻¹. Repare: $a_x$ depende de $v_y$ (por causa do $|\vec{v}|$) e vice-versa. Os dois movimentos ficam
# acoplados, e por isso não existe uma fórmula fechada para o alcance.

# %% [markdown]
# ## 4. O método de Euler: simular em passos pequenos
#
# Ideia: num intervalo de tempo bem curto, $\Delta t$, a velocidade e a aceleração quase não mudam. Então, a cada passo,
# fingimos que são constantes:
#
# $$ x \leftarrow x + v_x \,\Delta t \qquad y \leftarrow y + v_y \,\Delta t \qquad v_x \leftarrow v_x + a_x \,\Delta t \qquad v_y \leftarrow v_y + a_y \,\Delta t $$
#
# Repetimos até a bola tocar o chão. Quanto menor $\Delta t$, mais preciso (e mais demorado). Esse é o método de Euler
# (Euler method), o mais simples para resolver equações de movimento no computador.
#
# Repare na semelhança com o gradiente descendente: lá, pequenos passos contra a inclinação; aqui, pequenos passos no tempo.
#
# **Exemplo à mão**, um passo com $\Delta t = 0{,}1$ s, partindo da origem com $v_x = v_y = 10$ m/s, **sem** ar ($c = 0$):
#
# - $x = 0 + 10 \cdot 0{,}1 = 1$ m, $\quad y = 0 + 10 \cdot 0{,}1 = 1$ m;
# - $v_x = 10 + 0 = 10$ m/s, $\quad v_y = 10 + (-9{,}8)(0{,}1) = 9{,}02$ m/s.
#
# **O pouso.** O último passo quase sempre "passa do chão" (termina com $y < 0$). Para achar o ponto de pouso, ligamos os dois
# últimos pontos por uma reta e vemos onde ela cruza $y = 0$ (interpolação linear, linear interpolation).

# %%
def simular_trajetoria(v0, angulo_graus, c=0.013, dt=0.01):
    """Simula um lançamento pelo método de Euler. Devolve as listas de x e y até o pouso."""
    angulo = np.radians(angulo_graus)
    x, y = 0.0, 0.0
    vx, vy = v0 * np.cos(angulo), v0 * np.sin(angulo)
    lista_x, lista_y = [x], [y]
    while y >= 0:
        modulo_v = np.sqrt(vx**2 + vy**2)
        ax = -c * modulo_v * vx
        ay = -g - c * modulo_v * vy
        x, y = x + vx * dt, y + vy * dt
        vx, vy = vx + ax * dt, vy + ay * dt
        lista_x.append(x)
        lista_y.append(y)
    # interpolação linear entre o último ponto acima do chão e o primeiro abaixo
    x_antes, y_antes = lista_x[-2], lista_y[-2]
    lista_x[-1] = x_antes + (x - x_antes) * y_antes / (y_antes - y)
    lista_y[-1] = 0.0
    return np.array(lista_x), np.array(lista_y)


# Conferência: sem ar (c = 0), a simulação deve bater com a fórmula
x_vacuo, _ = simular_trajetoria(20, 45, c=0.0)
print(f"Euler sem ar: {x_vacuo[-1]:.2f} m   fórmula: {alcance_sem_ar(20, 45):.2f} m")

# %% [markdown]
# A diferença é de uns 14 cm em 41 m (0,3%), causada pelo passo de tempo finito. O Exercício 2 mostra que ela diminui com
# $\Delta t$ menor. A conferência nos dá confiança para ligar o ar.

# %%
fig, (eixo_traj, eixo_ang) = plt.subplots(1, 2, figsize=(13, 4.5))
for angulo in [30, 45, 60]:
    x_sem, y_sem = simular_trajetoria(30, angulo, c=0.0)
    x_com, y_com = simular_trajetoria(30, angulo)
    linha, = eixo_traj.plot(x_sem, y_sem, linestyle="--")
    eixo_traj.plot(x_com, y_com, color=linha.get_color(), label=f"{angulo}°")
eixo_traj.set_xlabel("x (m)")
eixo_traj.set_ylabel("y (m)")
eixo_traj.set_title("v0 = 30 m/s: tracejado sem ar, contínuo com ar")
eixo_traj.set_aspect("equal")
eixo_traj.legend()
eixo_traj.grid(True)

angulos = np.arange(5, 86, 1)
alcances_com_ar = [simular_trajetoria(30, a)[0][-1] for a in angulos]
eixo_ang.plot(angulos, alcance_sem_ar(30, angulos), "--", label="sem ar (fórmula)")
eixo_ang.plot(angulos, alcances_com_ar, label="com ar (Euler)")
eixo_ang.set_xlabel("ângulo (graus)")
eixo_ang.set_ylabel("alcance (m)")
eixo_ang.set_title(f"Ângulo ótimo com ar: {angulos[np.argmax(alcances_com_ar)]}°")
eixo_ang.legend()
eixo_ang.grid(True)
plt.show()

# %% [markdown]
# Com ar, o alcance cai muito, a trajetória deixa de ser simétrica (a descida é mais íngreme) e o melhor ângulo fica abaixo de 45°.

# %% [markdown]
# ## 5. Gerando os dados
#
# Vamos simular 2000 chutes com velocidades entre 10 e 35 m/s (até 126 km/h) e ângulos entre 10° e 70°.
# Para ser rápido, simulamos **todos os chutes ao mesmo tempo**, com arrays (a vetorização do Módulo 02): cada posição dos arrays
# é um chute diferente. Uma máscara booleana (Módulo 00) marca quais bolas ainda estão no ar.

# %%
def simular_alcances(velocidades, angulos_graus, c=0.013, dt=0.01):
    """Método de Euler para muitos lançamentos de uma vez. Devolve o alcance de cada um."""
    angulos = np.radians(angulos_graus)
    x, y = np.zeros(len(velocidades)), np.zeros(len(velocidades))
    vx, vy = velocidades * np.cos(angulos), velocidades * np.sin(angulos)
    alcances = np.full(len(velocidades), np.nan)
    no_ar = np.ones(len(velocidades), dtype=bool)
    while no_ar.any():
        modulo_v = np.sqrt(vx**2 + vy**2)
        ax = -c * modulo_v * vx
        ay = -g - c * modulo_v * vy
        x_novo, y_novo = x + vx * dt, y + vy * dt
        pousou = no_ar & (y_novo < 0)
        alcances[pousou] = x[pousou] + (x_novo[pousou] - x[pousou]) * y[pousou] / (y[pousou] - y_novo[pousou])
        no_ar = no_ar & ~pousou
        x, y = x_novo, y_novo
        vx, vy = vx + ax * dt, vy + ay * dt
    return alcances


rng = np.random.default_rng(42)
n_chutes = 2000
velocidades = rng.uniform(10, 35, n_chutes)
angulos_chute = rng.uniform(10, 70, n_chutes)

inicio = time.perf_counter()
alcances = simular_alcances(velocidades, angulos_chute)
print(f"{n_chutes} simulações em {time.perf_counter() - inicio:.2f} s")
print(f"conferência com a versão de um chute: {simular_alcances(np.array([30.0]), np.array([45.0]))[0]:.3f} m "
      f"e {simular_trajetoria(30, 45)[0][-1]:.3f} m")

# %%
plt.figure(figsize=(7, 4.5))
pontos = plt.scatter(angulos_chute, velocidades, c=alcances, s=8, cmap="viridis")
plt.colorbar(pontos, label="alcance com ar (m)")
plt.xlabel("ângulo (graus)")
plt.ylabel("velocidade inicial (m/s)")
plt.title("2000 chutes simulados")
plt.show()

# %% [markdown]
# Agora o problema virou um problema de aprendizado: entradas (velocidade, ângulo), alvo (alcance).
# Separamos treino, validação e teste (Módulo 05) e normalizamos (Módulo 02). Desta vez normalizamos também o **alvo**:
# alcances vão de poucos metros a quase 70, e a rede aprende melhor com números em torno de 0 e 1. Depois desfazemos a conta.

# %%
X_todos = np.column_stack([velocidades, angulos_chute])
y_todos = alcances.reshape(-1, 1)

ordem = rng.permutation(n_chutes)
indices_treino, indices_validacao, indices_teste = ordem[:1200], ordem[1200:1600], ordem[1600:]

media_X, desvio_X = X_todos[indices_treino].mean(axis=0), X_todos[indices_treino].std(axis=0)
media_y, desvio_y = y_todos[indices_treino].mean(), y_todos[indices_treino].std()


def preparar_entradas(X):
    """Normaliza as entradas com a média e o desvio do treino."""
    return (X - media_X) / desvio_X


X_treino, y_treino = preparar_entradas(X_todos[indices_treino]), (y_todos[indices_treino] - media_y) / desvio_y
X_validacao, y_validacao = preparar_entradas(X_todos[indices_validacao]), (y_todos[indices_validacao] - media_y) / desvio_y
alcances_teste = y_todos[indices_teste]
print(f"treino: {len(X_treino)}   validação: {len(X_validacao)}   teste: {len(indices_teste)}")

# %% [markdown]
# Um detalhe importante: a média e o desvio vêm **só do treino**. A validação e o teste fazem o papel de dados novos, que
# chegariam depois; não podemos "espiar" neles nem para normalizar.

# %% [markdown]
# ## 6. A rede para prever um número
#
# É a rede do Módulo 05 (uma camada escondida com ReLU, inicialização de He), com duas mudanças para prever um número
# em vez de uma classe:
#
# - **saída linear:** sem sigmoide nem softmax; a saída é a própria soma $\hat{y} = H W_2 + b_2$, que pode ser qualquer número;
# - **perda:** o erro quadrático médio do Módulo 01, $L = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2$.
#
# Na retropropagação, só o primeiro elo muda: a derivada do erro quadrático em relação à saída é $\frac{2}{n}(\hat{y} - y)$
# (a mesma conta do Módulo 01). Do segundo elo em diante, tudo é igual ao Módulo 04.

# %%
def relu(z):
    """Vimos isso no Módulo 04."""
    return np.maximum(0, z)


def derivada_relu(z):
    """Vimos isso no Módulo 04."""
    return (z > 0).astype(float)


def inicializar_rede(n_entradas, n_ocultos, n_saidas, semente=0):
    """Vimos isso no Módulo 05: inicialização de He."""
    gerador = np.random.default_rng(semente)
    return {
        "W1": gerador.normal(0, np.sqrt(2 / n_entradas), (n_entradas, n_ocultos)),
        "b1": np.zeros(n_ocultos),
        "W2": gerador.normal(0, np.sqrt(2 / n_ocultos), (n_ocultos, n_saidas)),
        "b2": np.zeros(n_saidas),
    }


def propagacao_frente(X, rede):
    """Camada escondida ReLU e saída linear (sem ativação)."""
    Z1 = X @ rede["W1"] + rede["b1"]
    H = relu(Z1)
    saida = H @ rede["W2"] + rede["b2"]
    return saida, {"X": X, "Z1": Z1, "H": H, "saida": saida}


def erro_quadratico_medio(previsto, medido):
    """Vimos isso no Módulo 01."""
    return np.mean((previsto - medido) ** 2)


def retropropagacao(y, rede, memoria):
    """Como no Módulo 04; só o primeiro elo muda: 2 (previsto - real) / n."""
    n = len(y)
    dZ2 = 2 * (memoria["saida"] - y) / n
    dW2 = memoria["H"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dH = dZ2 @ rede["W2"].T
    dZ1 = dH * derivada_relu(memoria["Z1"])
    dW1 = memoria["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

# %% [markdown]
# Mudamos um elo, então conferimos os gradientes de novo (Módulo 04), numa rede pequena e com poucos exemplos.

# %%
def conferir_gradientes(X, y, rede, passinho=1e-5):
    """Vimos isso no Módulo 04, agora com o erro quadrático médio."""
    analiticos = retropropagacao(y, rede, propagacao_frente(X, rede)[1])
    for nome in rede:
        numericos = np.zeros_like(rede[nome])
        for posicao in np.ndindex(rede[nome].shape):
            original = rede[nome][posicao]
            rede[nome][posicao] = original + passinho
            perda_mais = erro_quadratico_medio(propagacao_frente(X, rede)[0], y)
            rede[nome][posicao] = original - passinho
            perda_menos = erro_quadratico_medio(propagacao_frente(X, rede)[0], y)
            rede[nome][posicao] = original
            numericos[posicao] = (perda_mais - perda_menos) / (2 * passinho)
        print(f"{nome}: maior diferença = {np.max(np.abs(numericos - analiticos[nome])):.2e}")


conferir_gradientes(X_treino[:10], y_treino[:10], inicializar_rede(2, 5, 1))

# %% [markdown]
# ## 7. Treinando
#
# Minilotes de 32, como no Módulo 05; 64 neurônios escondidos, taxa 0,05, 300 épocas.

# %%
def treinar(X, y, X_val, y_val, n_ocultos, taxa, epocas, tamanho_lote=32, semente=0):
    """SGD com minilotes para regressão. Guarda o erro quadrático médio de treino e de validação por época."""
    rede = inicializar_rede(X.shape[1], n_ocultos, 1, semente)
    gerador = np.random.default_rng(semente)
    historico = {"treino": [], "validacao": []}
    for epoca in range(epocas):
        ordem_lotes = gerador.permutation(len(X))
        for inicio in range(0, len(X), tamanho_lote):
            lote = ordem_lotes[inicio:inicio + tamanho_lote]
            _, memoria = propagacao_frente(X[lote], rede)
            gradientes = retropropagacao(y[lote], rede, memoria)
            for nome in rede:
                rede[nome] = rede[nome] - taxa * gradientes[nome]
        historico["treino"].append(erro_quadratico_medio(propagacao_frente(X, rede)[0], y))
        historico["validacao"].append(erro_quadratico_medio(propagacao_frente(X_val, rede)[0], y_val))
    return rede, historico


def prever_alcance(rede, velocidades, angulos_graus):
    """Alcance previsto pela rede, em metros (normaliza a entrada e desfaz a normalização da saída)."""
    X = preparar_entradas(np.column_stack([velocidades, angulos_graus]))
    return (propagacao_frente(X, rede)[0] * desvio_y + media_y).ravel()


inicio = time.perf_counter()
rede, historico = treinar(X_treino, y_treino, X_validacao, y_validacao, n_ocultos=64, taxa=0.05, epocas=300)
print(f"treino em {time.perf_counter() - inicio:.1f} s")

# %% [markdown]
# **O aprendizado acontecendo.** O erro está em unidades normalizadas (multiplique a raiz por $\sigma_y$ para ter metros).

# %%
plt.figure(figsize=(7, 4))
plt.plot(historico["treino"], label="treino")
plt.plot(historico["validacao"], label="validação")
plt.yscale("log")
plt.xlabel("época")
plt.ylabel("erro quadrático médio (normalizado)")
plt.title("A rede aprendendo a física do chute")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 8. Rede contra simulação
#
# Agora a prova: o conjunto de teste, que a rede nunca viu. Medimos o erro médio absoluto (mean absolute error), a média de
# $|\text{previsto} - \text{real}|$, em metros: é mais fácil de interpretar do que o erro quadrático.

# %%
previsto_teste = prever_alcance(rede, X_todos[indices_teste, 0], X_todos[indices_teste, 1])
real_teste = alcances_teste.ravel()
erro_medio_absoluto = np.mean(np.abs(previsto_teste - real_teste))
print(f"erro médio absoluto no teste: {erro_medio_absoluto:.2f} m")
print(f"maior erro no teste: {np.max(np.abs(previsto_teste - real_teste)):.2f} m")

fig, (eixo_disp, eixo_curva) = plt.subplots(1, 2, figsize=(13, 4.5))
eixo_disp.scatter(real_teste, previsto_teste, s=8)
eixo_disp.plot([0, real_teste.max()], [0, real_teste.max()], color="black", linestyle="--", label="previsão perfeita")
eixo_disp.set_xlabel("alcance simulado (m)")
eixo_disp.set_ylabel("alcance previsto pela rede (m)")
eixo_disp.set_title(f"Teste: erro médio de {erro_medio_absoluto:.2f} m")
eixo_disp.legend()
eixo_disp.grid(True)

angulos_finos = np.linspace(10, 70, 121)
for v0 in [15, 25, 35]:
    simulado = simular_alcances(np.full(len(angulos_finos), float(v0)), angulos_finos)
    previsto = prever_alcance(rede, np.full(len(angulos_finos), float(v0)), angulos_finos)
    linha, = eixo_curva.plot(angulos_finos, simulado, label=f"simulação, v0 = {v0} m/s")
    eixo_curva.plot(angulos_finos, previsto, "--", color=linha.get_color(),
                    label=f"rede (ótimo em {angulos_finos[np.argmax(previsto)]:.1f}°)")
eixo_curva.set_xlabel("ângulo (graus)")
eixo_curva.set_ylabel("alcance (m)")
eixo_curva.legend(fontsize=8)
eixo_curva.grid(True)
plt.show()

# %% [markdown]
# Os pontos ficam colados na diagonal, e as curvas da rede (tracejadas) quase cobrem as da simulação, inclusive o ângulo ótimo.
# A rede nunca viu a equação do arrasto: aprendeu só com exemplos.

# %% [markdown]
# ## 9. Por que usar uma rede, se temos o simulador?
#
# **Velocidade.** Depois de treinada, a rede responde sempre com a mesma conta curta: duas multiplicações de matriz.
# O simulador precisa de centenas de passos de Euler por chute, e de dez vezes mais passos se quisermos uma casa decimal a mais
# de precisão (passo $\Delta t$ dez vezes menor; veja o Exercício 2). Vamos medir os tempos para 20 mil chutes.

# %%
muitas_velocidades = rng.uniform(10, 35, 20000)
muitos_angulos = rng.uniform(10, 70, 20000)

for dt in [0.01, 0.001]:
    inicio = time.perf_counter()
    simular_alcances(muitas_velocidades, muitos_angulos, dt=dt)
    print(f"simulação com dt = {dt}: {time.perf_counter() - inicio:.3f} s")

inicio = time.perf_counter()
prever_alcance(rede, muitas_velocidades, muitos_angulos)
print(f"rede:                    {time.perf_counter() - inicio:.3f} s")

# %% [markdown]
# Os tempos variam de máquina para máquina. Aqui o simulador também é rápido, porque a física é simples e simulamos todos os chutes
# ao mesmo tempo; a diferença cresce com a precisão pedida. Agora pense numa simulação de verdade: o ar em volta de uma asa,
# o clima do planeta, o enovelamento de uma proteína. Uma única rodada pode levar horas num supercomputador. Uma rede treinada
# com rodadas antigas pode dar uma resposta aproximada em milissegundos. Esse uso tem nome: modelo substituto (surrogate model).

# %% [markdown]
# ## 10. O limite: fora da faixa de treino
#
# A rede só viu velocidades entre 10 e 35 m/s. O que acontece se pedirmos um chute de 50 m/s? Prever fora da faixa dos dados de
# treino é extrapolar (extrapolation).

# %%
velocidades_largas = np.linspace(2, 60, 200)
angulo_fixo = np.full(len(velocidades_largas), 40.0)
simulado_largo = simular_alcances(velocidades_largas, angulo_fixo)
previsto_largo = prever_alcance(rede, velocidades_largas, angulo_fixo)

plt.figure(figsize=(8, 4.5))
plt.axvspan(10, 35, color="green", alpha=0.1, label="faixa de treino")
plt.plot(velocidades_largas, simulado_largo, label="simulação (com ar)")
plt.plot(velocidades_largas, previsto_largo, "--", label="rede")
plt.plot(velocidades_largas, alcance_sem_ar(velocidades_largas, 40), ":", color="gray", label="fórmula sem ar")
plt.ylim(-5, 120)
plt.xlabel("velocidade inicial (m/s)")
plt.ylabel("alcance (m)")
plt.title("Ângulo de 40°: dentro e fora da faixa de treino")
plt.legend()
plt.grid(True)
plt.show()

for v0 in [30, 45, 60]:
    print(f"v0 = {v0} m/s: simulação {simular_alcances(np.array([float(v0)]), np.array([40.0]))[0]:6.1f} m"
          f"   rede {prever_alcance(rede, np.array([float(v0)]), np.array([40.0]))[0]:6.1f} m")

# %% [markdown]
# Dentro da faixa verde, a rede acerta. Fora dela, se afasta da simulação, cada vez mais. Uma rede ReLU, longe dos dados,
# simplesmente continua em linha reta: ela não "sabe" Física, só imita os exemplos que viu. Já o simulador vale para qualquer
# velocidade, porque foi construído a partir das leis de Newton.
#
# ### Quando faz sentido usar IA em Física?
#
# **Faz sentido** quando:
#
# - a simulação ou o experimento é caro e precisamos de muitas respostas rápidas (otimização, controle em tempo real);
# - há muitos dados, mas não conhecemos (ou não sabemos resolver) as equações;
# - uma resposta aproximada basta, e sabemos que as perguntas ficam dentro da faixa dos dados.
#
# **Cuidado** quando:
#
# - existe uma fórmula ou um simulador barato: eles são exatos, explicáveis e valem em qualquer faixa;
# - é preciso extrapolar, ou seja, responder sobre situações diferentes das que a rede viu;
# - a resposta precisa respeitar leis de conservação (energia, momento): a rede não as garante, a menos que sejam embutidas nela.
#
# Uma linha de pesquisa atual combina as duas coisas: redes que recebem a Física como ajuda (atributos, restrições, equações).
# O Exercício 3 dá um gostinho disso.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Faça à mão **um passo** do método de Euler **com** ar: partindo da origem, $v_x = v_y = 10$ m/s,
# $c = 0{,}01$ m⁻¹, $\Delta t = 0{,}1$ s. Calcule $|\vec{v}|$, $a_x$, $a_y$ e depois os novos $x$, $y$, $v_x$, $v_y$.
# Confira com algumas linhas de código.
#
# **Exercício 2 (médio).** Compare o alcance do método de Euler **sem** ar ($c = 0$), para $v_0 = 30$ m/s e $45°$, com a fórmula,
# usando $\Delta t$ = 0,1; 0,01 e 0,001 s. Mostre o erro e o tempo de cada simulação (use `simular_alcances` com 1000 cópias do mesmo
# chute para o tempo ficar mensurável). O que acontece com o erro quando $\Delta t$ cai 10 vezes?
#
# **Exercício 3 (desafio).** Dê à rede uma "dica" de Física: acrescente uma terceira entrada, o alcance **sem ar**
# $v_0^2 \sin(2\theta)/g$. Treine com os mesmos dados e hiperparâmetros (normalize a nova coluna também, com a média e o desvio do treino)
# e compare o erro no teste e o comportamento fora da faixa de treino (o gráfico da seção 10).

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | projectile | projétil | *The ball is a projectile after the kick.* |
# | range | alcance | *The range is maximum at 45 degrees in a vacuum.* |
# | air resistance / drag | resistência do ar / arrasto | *Drag grows with the square of the speed.* |
# | time step | passo de tempo | *A smaller time step gives a more accurate simulation.* |
# | Euler method | método de Euler | *The Euler method updates position and velocity step by step.* |
# | simulation | simulação | *Each simulation takes hundreds of steps.* |
# | linear interpolation | interpolação linear | *Linear interpolation finds the landing point.* |
# | regression | regressão | *Predicting the range is a regression problem.* |
# | mean absolute error | erro médio absoluto | *The mean absolute error is a few centimeters.* |
# | surrogate model | modelo substituto | *A surrogate model replaces an expensive simulation.* |
# | extrapolation | extrapolação | *Neural networks are bad at extrapolation.* |
# | interpolation | interpolação | *Inside the training range, the network interpolates well.* |
# | conservation law | lei de conservação | *A network does not know conservation laws by itself.* |

# %% [markdown]
# ## Resumo
#
# - Sem ar, o alcance tem fórmula, $v_0^2 \sin(2\theta)/g$; com arrasto quadrático, não há fórmula simples.
# - O método de Euler simula o movimento em passos pequenos de tempo; sem ar, confere com a fórmula.
# - Com dados gerados pela simulação, a mesma rede dos módulos anteriores (com saída linear e erro quadrático) aprende a prever o alcance.
# - A rede é muito rápida depois de treinada: serve de modelo substituto para simulações caras.
# - Fora da faixa de treino, a rede erra feio: ela imita exemplos, não conhece as leis da Física.
#
# **Próximo módulo:** o PyTorch, uma biblioteca que calcula os gradientes sozinha. Você vai conferir que ela faz o mesmo que você fez à mão.

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
# - $|\vec{v}| = \sqrt{10^2 + 10^2} \approx 14{,}14$ m/s
# - $a_x = -0{,}01 \cdot 14{,}14 \cdot 10 \approx -1{,}414$ m/s², $\quad a_y = -9{,}8 - 1{,}414 = -11{,}214$ m/s²
# - $x = 1$ m, $\; y = 1$ m, $\; v_x = 10 - 0{,}1414 \approx 9{,}859$ m/s, $\; v_y = 10 - 1{,}1214 \approx 8{,}879$ m/s

# %%
vx, vy, c_ex, dt_ex = 10.0, 10.0, 0.01, 0.1
modulo = np.sqrt(vx**2 + vy**2)
ax, ay = -c_ex * modulo * vx, -g - c_ex * modulo * vy
print(f"|v| = {modulo:.3f}   ax = {ax:.3f}   ay = {ay:.3f}")
print(f"x = {vx * dt_ex:.3f}   y = {vy * dt_ex:.3f}   vx = {vx + ax * dt_ex:.3f}   vy = {vy + ay * dt_ex:.3f}")

# %% [markdown]
# ### Solução do Exercício 2

# %%
copias_v, copias_angulo = np.full(1000, 30.0), np.full(1000, 45.0)
for dt in [0.1, 0.01, 0.001]:
    inicio = time.perf_counter()
    alcance_euler = simular_alcances(copias_v, copias_angulo, c=0.0, dt=dt)[0]
    duracao = time.perf_counter() - inicio
    print(f"dt = {dt}: alcance {alcance_euler:.4f} m   erro {abs(alcance_euler - alcance_sem_ar(30, 45)):.4f} m   tempo {duracao:.3f} s")

# %% [markdown]
# Quando $\Delta t$ cai 10 vezes, o erro cai cerca de 10 vezes e o tempo sobe cerca de 10 vezes. O erro do método de Euler é
# proporcional ao passo: dizemos que ele é um método de primeira ordem. Métodos mais sofisticados (como Runge-Kutta) erram muito
# menos com o mesmo passo.

# %% [markdown]
# ### Solução do Exercício 3

# %%
X_fisica = np.column_stack([velocidades, angulos_chute, alcance_sem_ar(velocidades, angulos_chute)])
media_f, desvio_f = X_fisica[indices_treino].mean(axis=0), X_fisica[indices_treino].std(axis=0)


def entradas_com_fisica(v, angulo):
    """Velocidade, ângulo e alcance sem ar, normalizados com as estatísticas do treino."""
    return (np.column_stack([v, angulo, alcance_sem_ar(v, angulo)]) - media_f) / desvio_f


rede_fisica, _ = treinar(entradas_com_fisica(velocidades[indices_treino], angulos_chute[indices_treino]), y_treino,
                         entradas_com_fisica(velocidades[indices_validacao], angulos_chute[indices_validacao]), y_validacao,
                         n_ocultos=64, taxa=0.05, epocas=300)


def prever_com_fisica(v, angulo):
    """Alcance previsto (m) pela rede com a dica de Física."""
    return (propagacao_frente(entradas_com_fisica(v, angulo), rede_fisica)[0] * desvio_y + media_y).ravel()


previsto_fisica = prever_com_fisica(velocidades[indices_teste], angulos_chute[indices_teste])
print(f"erro médio no teste: sem dica {erro_medio_absoluto:.2f} m   com dica {np.mean(np.abs(previsto_fisica - real_teste)):.2f} m")

plt.figure(figsize=(8, 4.5))
plt.axvspan(10, 35, color="green", alpha=0.1, label="faixa de treino")
plt.plot(velocidades_largas, simulado_largo, label="simulação (com ar)")
plt.plot(velocidades_largas, previsto_largo, "--", label="rede sem dica")
plt.plot(velocidades_largas, prever_com_fisica(velocidades_largas, angulo_fixo), "-.", label="rede com dica de Física")
plt.ylim(-5, 120)
plt.xlabel("velocidade inicial (m/s)")
plt.ylabel("alcance (m)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Dentro da faixa de treino, a dica ajuda: o erro médio no teste cai (aqui, mais ou menos pela metade). Fora da faixa, a dica muda
# o comportamento: como o alcance sem ar cresce
# com $v_0^2$, a rede que o recebe tende a extrapolar **para cima**, enquanto o alcance real com ar cresce cada vez mais devagar.
# A dica certa ajuda; a dica errada pode atrapalhar fora da faixa. Por isso, extrapolar continua arriscado mesmo com ajuda da Física.
