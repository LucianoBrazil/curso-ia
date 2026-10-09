# %% [markdown]
# # Módulo B1 (bônus) — O pêndulo: uma rede que aprende a dinâmica
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B1-pendulo.ipynb)
#
# **Objetivo.** Ensinar uma rede neural a prever como um pêndulo se move, um pequeno passo de tempo de cada vez, e descobrir o que acontece
# quando usamos a própria previsão como ponto de partida da previsão seguinte: o erro se acumula.
#
# **O que você vai construir.** Um simulador de pêndulo (e um teste de energia que revela um defeito do método de Euler), uma rede em PyTorch
# que aprende o "próximo estado" do pêndulo e uma simulação de 20 segundos feita só pela rede, comparada com a Física.

# %% [markdown]
# ## 1. A intuição: prever o futuro passo a passo
#
# No Módulo 06, a rede respondia uma pergunta única: "até onde vai a bola?". Muitos problemas são diferentes: queremos o **filme inteiro**.
# Onde estará o pêndulo daqui a 0,05 s? E daqui a 20 s?
#
# A estratégia natural é aprender a regra de um passo, "estado agora → estado daqui a pouco", e aplicá-la de novo e de novo, como o método
# de Euler. É assim que se faz previsão do tempo, simulação de robôs e, como no Módulo 10, geração de texto token por token.
#
# O perigo: cada passo usa o resultado do passo anterior. Um erro minúsculo vira o ponto de partida do passo seguinte, e os erros vão se
# somando. Este módulo mede esse efeito.

# %% [markdown]
# ## 2. A Física do pêndulo
#
# Um pêndulo simples: uma massa $m$ presa a uma haste sem massa de comprimento $L$. O ângulo com a vertical é $\theta$.
# A componente do peso ao longo da trajetória é $-m g \sin\theta$ (sempre puxando de volta para baixo). Pela segunda lei de Newton,
# com a aceleração ao longo do arco valendo $L\,\ddot\theta$:
#
# $$ m L\, \ddot{\theta} = -m g \sin\theta \quad\Rightarrow\quad \ddot{\theta} = -\frac{g}{L}\sin\theta $$
#
# ($\ddot\theta$ é a aceleração angular, a "taxa de variação" da velocidade angular $\omega$.) O estado do pêndulo num instante é o par
# ($\theta$, $\omega$): sabendo os dois, a equação determina todo o futuro.
#
# **Ângulos pequenos.** Para $\theta$ pequeno (em radianos), $\sin\theta \approx \theta$, e o pêndulo vira um oscilador harmônico com período
#
# $$ T = 2\pi\sqrt{L/g} $$
#
# **Exemplo à mão.** $L = 1$ m: $\sqrt{1/9{,}8} \approx 0{,}3194$, e $T \approx 2\pi \cdot 0{,}3194 \approx 2{,}007$ s.
#
# Para ângulos grandes, $\sin\theta$ não é mais parecido com $\theta$, e não existe uma fórmula simples para o movimento. Vamos simular.
#
# **Energia.** Sem atrito, a energia mecânica se conserva. Por unidade de massa, com $L = 1$ m:
#
# $$ E = \tfrac{1}{2}\,\omega^2 + g\,(1 - \cos\theta) $$
#
# (cinética mais potencial, medindo a altura a partir do ponto mais baixo). Exemplo: solto do repouso em $\theta = 0{,}5$ rad,
# $E = 9{,}8\,(1 - \cos 0{,}5) \approx 9{,}8 \cdot 0{,}1224 \approx 1{,}20$ J/kg. A energia será o nosso "detector de mentiras".

# %% [markdown]
# ## 3. Simulando: o método de Euler e um defeito escondido
#
# O método de Euler do Módulo 06, aplicado ao pêndulo, com passo $\Delta t$:
#
# $$ \theta_{\text{novo}} = \theta + \omega\,\Delta t \qquad\qquad \omega_{\text{novo}} = \omega - \tfrac{g}{L}\sin\theta\;\Delta t $$
#
# Uma variação mínima, o Euler semi-implícito (semi-implicit Euler, também chamado de Euler-Cromer), atualiza primeiro a velocidade e usa a
# velocidade **nova** para atualizar o ângulo:
#
# $$ \omega_{\text{novo}} = \omega - \tfrac{g}{L}\sin\theta\;\Delta t \qquad\qquad \theta_{\text{novo}} = \theta + \omega_{\text{novo}}\,\Delta t $$
#
# **Exemplo à mão**, um passo de $\Delta t = 0{,}1$ s a partir de $\theta = 0{,}5$ rad, $\omega = 0$ (com $\sin 0{,}5 \approx 0{,}479$):
#
# - nos dois métodos, $\omega_{\text{novo}} = 0 - 9{,}8 \cdot 0{,}479 \cdot 0{,}1 \approx -0{,}470$ rad/s;
# - Euler: $\theta_{\text{novo}} = 0{,}5 + 0 \cdot 0{,}1 = 0{,}5$ (usou a velocidade antiga, zero);
# - semi-implícito: $\theta_{\text{novo}} = 0{,}5 + (-0{,}470)(0{,}1) \approx 0{,}453$.
#
# Parece um detalhe. Vamos ver a energia ao longo de 20 s.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

g, L = 9.8, 1.0


def energia(theta, omega):
    """Energia mecânica por unidade de massa (J/kg), com L = 1 m."""
    return 0.5 * omega**2 + g * (1 - np.cos(theta))


def simular(theta0, omega0, duracao, dt, metodo="semi-implicito"):
    """Simula o pêndulo. Devolve os arrays de tempo, ângulo e velocidade angular."""
    n_passos = int(round(duracao / dt))
    theta, omega = np.zeros(n_passos + 1), np.zeros(n_passos + 1)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_passos):
        aceleracao = -(g / L) * np.sin(theta[i])
        if metodo == "euler":
            theta[i + 1] = theta[i] + omega[i] * dt
            omega[i + 1] = omega[i] + aceleracao * dt
        else:
            omega[i + 1] = omega[i] + aceleracao * dt
            theta[i + 1] = theta[i] + omega[i + 1] * dt
    return np.arange(n_passos + 1) * dt, theta, omega


print(f"energia inicial (θ = 0,5 rad, ω = 0): {energia(0.5, 0.0):.3f} J/kg")
_, theta_um_passo, omega_um_passo = simular(0.5, 0.0, 0.1, 0.1)
print(f"um passo semi-implícito: θ = {theta_um_passo[1]:.3f}, ω = {omega_um_passo[1]:.3f}")

fig, (eixo_theta, eixo_energia) = plt.subplots(1, 2, figsize=(13, 4))
for metodo, estilo in [("euler", "-"), ("semi-implicito", "--")]:
    t, theta, omega = simular(1.0, 0.0, 20, 0.01, metodo)
    eixo_theta.plot(t, theta, estilo, label=metodo)
    eixo_energia.plot(t, energia(theta, omega), estilo, label=metodo)
eixo_theta.set_xlabel("tempo (s)")
eixo_theta.set_ylabel("θ (rad)")
eixo_theta.set_title("Solto de 1 rad, passo de 0,01 s")
eixo_theta.legend()
eixo_theta.grid(True)
eixo_energia.set_xlabel("tempo (s)")
eixo_energia.set_ylabel("energia (J/kg)")
eixo_energia.set_title("A energia deveria ficar constante")
eixo_energia.legend()
eixo_energia.grid(True)
plt.show()

# %% [markdown]
# Com o Euler simples, a energia **cresce** sem parar: o pêndulo balança cada vez mais alto, como se alguém o empurrasse. É o erro de cada
# passo se acumulando, sempre no mesmo sentido. Com o semi-implícito, a energia oscila um pouquinho, mas não foge. Métodos que respeitam a
# estrutura da Física desse jeito se chamam simpléticos (symplectic); são os preferidos para simular planetas e moléculas.
#
# É a primeira lição do módulo: **pequenos erros por passo podem se acumular**. Vamos ver se uma rede neural escapa disso.
#
# A partir daqui, a "verdade" é o semi-implícito com passo bem pequeno, $\Delta t = 0{,}001$ s. O passo da rede será 50 vezes maior: 0,05 s.

# %%
DT_REDE = 0.05      # passo de tempo que a rede vai aprender
DT_FINO = 0.001     # passo da simulação de referência


def passo_verdadeiro(theta, omega):
    """Avança arrays de estados em DT_REDE segundos, com 50 passos semi-implícitos de DT_FINO."""
    for _ in range(int(round(DT_REDE / DT_FINO))):
        omega = omega - (g / L) * np.sin(theta) * DT_FINO
        theta = theta + omega * DT_FINO
    return theta, omega


def trajetoria_verdadeira(theta0, omega0, n_passos):
    """Estados a cada DT_REDE segundos, pela simulação de referência."""
    estados = [(theta0, omega0)]
    theta, omega = np.array([theta0]), np.array([omega0])
    for _ in range(n_passos):
        theta, omega = passo_verdadeiro(theta, omega)
        estados.append((theta[0], omega[0]))
    return np.array(estados)

# %% [markdown]
# ## 4. Os dados: milhares de "antes e depois"
#
# Sorteamos 4000 estados (ângulos entre −2,5 e 2,5 rad, velocidades entre −6 e 6 rad/s) e, para cada um, calculamos o estado 0,05 s depois.
# A rede vai aprender essa regra.
#
# **Um truque.** Em vez de prever o estado novo, a rede prevê a **variação**: $\Delta\theta$ e $\Delta\omega$. Em 0,05 s o pêndulo quase não
# muda; prever "quase o mesmo número" é difícil de fazer com precisão, prever "a pequena mudança" é mais fácil. É a mesma ideia das conexões
# residuais do Módulo 10: $\text{novo} = \text{atual} + \text{ajuste}$.

# %%
rng = np.random.default_rng(42)
n_exemplos = 4000
theta_inicial = rng.uniform(-2.5, 2.5, n_exemplos)
omega_inicial = rng.uniform(-6, 6, n_exemplos)
theta_final, omega_final = passo_verdadeiro(theta_inicial, omega_inicial)

entradas = np.column_stack([theta_inicial, omega_inicial])
variacoes = np.column_stack([theta_final - theta_inicial, omega_final - omega_inicial])

# normalização (Módulo 02), com as estatísticas do treino
indices = rng.permutation(n_exemplos)
indices_treino, indices_teste = indices[:3200], indices[3200:]
media_entrada, desvio_entrada = entradas[indices_treino].mean(axis=0), entradas[indices_treino].std(axis=0)
media_saida, desvio_saida = variacoes[indices_treino].mean(axis=0), variacoes[indices_treino].std(axis=0)


def para_tensor(array):
    """NumPy -> tensor float32."""
    return torch.tensor(array, dtype=torch.float32)


X_treino = para_tensor((entradas[indices_treino] - media_entrada) / desvio_entrada)
Y_treino = para_tensor((variacoes[indices_treino] - media_saida) / desvio_saida)
X_teste = para_tensor((entradas[indices_teste] - media_entrada) / desvio_entrada)
Y_teste = para_tensor((variacoes[indices_teste] - media_saida) / desvio_saida)
print("treino:", tuple(X_treino.shape), "  teste:", tuple(X_teste.shape))

# %% [markdown]
# ## 5. A rede e o treino
#
# Duas camadas escondidas de 64 neurônios com tanh (Módulo 09), saída linear com 2 números e erro quadrático médio (Módulo 06).
# Adam (Módulo 07), com a taxa reduzida no fim (Módulo 09). Como os dados são poucos e pequenos, cada passo usa o treino inteiro.

# %%
torch.manual_seed(0)
rede = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
otimizador = torch.optim.Adam(rede.parameters(), lr=0.003)
funcao_perda = nn.MSELoss()

passos = 3000
historico_treino, historico_teste = [], []
inicio = time.perf_counter()
for passo in range(passos):
    if passo == int(0.8 * passos):
        for grupo in otimizador.param_groups:
            grupo["lr"] = 0.0003
    perda = funcao_perda(rede(X_treino), Y_treino)
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()
    historico_treino.append(perda.item())
    with torch.no_grad():
        historico_teste.append(funcao_perda(rede(X_teste), Y_teste).item())
print(f"treino em {time.perf_counter() - inicio:.0f} s   erro quadrático no teste (normalizado): {historico_teste[-1]:.2e}")

# %% [markdown]
# **O aprendizado acontecendo.**

# %%
plt.figure(figsize=(7, 4))
plt.plot(historico_treino, label="treino")
plt.plot(historico_teste, label="teste")
plt.yscale("log")
plt.xlabel("passo")
plt.ylabel("erro quadrático médio (normalizado)")
plt.title("A rede aprendendo um passo da dinâmica")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Quão bom é um passo? Medimos o erro em unidades físicas, no conjunto de teste.

# %%
def passo_rede(theta, omega):
    """Avança arrays de estados em DT_REDE segundos usando a rede."""
    estado = (np.column_stack([theta, omega]) - media_entrada) / desvio_entrada
    with torch.no_grad():
        variacao = rede(para_tensor(estado)).numpy() * desvio_saida + media_saida
    return theta + variacao[:, 0], omega + variacao[:, 1]


theta_previsto, omega_previsto = passo_rede(theta_inicial[indices_teste], omega_inicial[indices_teste])
erro_theta = np.abs(theta_previsto - theta_final[indices_teste])
erro_omega = np.abs(omega_previsto - omega_final[indices_teste])
print(f"erro médio em um passo: θ {erro_theta.mean():.5f} rad   ω {erro_omega.mean():.5f} rad/s")
print(f"(em 0,05 s, θ muda em média {np.abs(variacoes[indices_teste, 0]).mean():.3f} rad)")

# %% [markdown]
# O erro de um passo é uma fração minúscula da própria mudança. Parece um ótimo modelo.

# %% [markdown]
# ## 6. O teste de verdade: 20 segundos só com a rede
#
# Agora a rede simula sozinha: parte de um estado inicial, prevê o próximo, e usa a **própria previsão** como entrada do passo seguinte.
# São 400 passos (20 s). Isso se chama simulação autorregressiva, ou rollout. Comparamos com a simulação de referência.

# %%
def trajetoria_rede(theta0, omega0, n_passos):
    """Estados a cada DT_REDE segundos, usando só a rede (cada passo parte da previsão anterior)."""
    estados = [(theta0, omega0)]
    theta, omega = np.array([theta0]), np.array([omega0])
    for _ in range(n_passos):
        theta, omega = passo_rede(theta, omega)
        estados.append((theta[0], omega[0]))
    return np.array(estados)


n_passos_rollout = 400
tempos = np.arange(n_passos_rollout + 1) * DT_REDE
verdade = trajetoria_verdadeira(1.0, 0.0, n_passos_rollout)
previsao = trajetoria_rede(1.0, 0.0, n_passos_rollout)

fig, eixos = plt.subplots(1, 3, figsize=(18, 4.5))
eixos[0].plot(tempos, verdade[:, 0], label="Física (referência)")
eixos[0].plot(tempos, previsao[:, 0], "--", label="rede")
eixos[0].set_xlabel("tempo (s)")
eixos[0].set_ylabel("θ (rad)")
eixos[0].set_title("Solto de 1 rad")
eixos[0].legend()
eixos[0].grid(True)

eixos[1].plot(tempos, np.abs(previsao[:, 0] - verdade[:, 0]) + 1e-6)
eixos[1].set_yscale("log")
eixos[1].set_xlabel("tempo (s)")
eixos[1].set_ylabel("|erro em θ| (rad)")
eixos[1].set_title("O erro se acumulando")
eixos[1].grid(True)

eixos[2].plot(tempos, energia(verdade[:, 0], verdade[:, 1]), label="Física (referência)")
eixos[2].plot(tempos, energia(previsao[:, 0], previsao[:, 1]), "--", label="rede")
eixos[2].set_xlabel("tempo (s)")
eixos[2].set_ylabel("energia (J/kg)")
eixos[2].set_title("A energia")
eixos[2].legend()
eixos[2].grid(True)
plt.show()

for segundos in [1, 5, 10, 20]:
    i = int(segundos / DT_REDE)
    print(f"após {segundos:2d} s: erro em θ = {abs(previsao[i, 0] - verdade[i, 0]):.4f} rad")

# %% [markdown]
# A olho nu, as curvas de θ parecem iguais. Mas o gráfico do meio, em escala logarítmica, conta outra história: o erro, que no primeiro
# segundo é de décimos de milésimo de radiano, cresce mais de 100 vezes em 20 s. A energia deveria ficar constante; na referência ela só
# oscila um pouquinho (um efeito do método semi-implícito), enquanto na rede ela sobe aos poucos. O erro de cada passo era minúsculo, mas
# foram 400 passos, cada um partindo de um ponto já um pouco errado.
#
# **Por que o erro cresce assim?** Se cada passo erra uns $\varepsilon$ e os erros não se cancelam, depois de $N$ passos o erro acumulado
# fica em torno de $N\varepsilon$, ou mais. Exemplo à mão: $\varepsilon = 0{,}001$ rad por passo, $N = 400$ passos, e o erro pode chegar a
# $0{,}4$ rad. No pêndulo há ainda um efeito que piora as coisas: um erro pequeno na energia muda um pouco o **período** da oscilação.
# Uma diferença de período, por menor que seja, faz a fase escorregar um pouco a cada oscilação, e esse atraso só cresce.
#
# Em sistemas caóticos, como o pêndulo duplo ou a atmosfera, é pior: erros pequenos crescem **exponencialmente**. É por isso que a previsão
# do tempo é boa para poucos dias, mesmo com os melhores modelos do mundo.
#
# E se a rede simular por mais tempo? Vamos deixá-la rodar 2 minutos (2400 passos) e olhar os últimos 10 segundos.

# %%
n_passos_longo = 2400
tempos_longos = np.arange(n_passos_longo + 1) * DT_REDE
verdade_longa = trajetoria_verdadeira(1.0, 0.0, n_passos_longo)
previsao_longa = trajetoria_rede(1.0, 0.0, n_passos_longo)
ultimos = tempos_longos >= 110

plt.figure(figsize=(9, 4))
plt.plot(tempos_longos[ultimos], verdade_longa[ultimos, 0], label="Física (referência)")
plt.plot(tempos_longos[ultimos], previsao_longa[ultimos, 0], "--", label="rede")
plt.xlabel("tempo (s)")
plt.ylabel("θ (rad)")
plt.title("Depois de quase 2 minutos simulando sozinha")
plt.legend()
plt.grid(True)
plt.show()

erro_longo = np.abs(previsao_longa[:, 0] - verdade_longa[:, 0])
for fim in [10, 30, 60, 120]:
    janela = (tempos_longos > fim - 10) & (tempos_longos <= fim)
    print(f"maior erro em θ entre {fim - 10:3d} e {fim:3d} s: {erro_longo[janela].max():.3f} rad")

# %% [markdown]
# Agora o atraso salta aos olhos: a oscilação da rede ficou fora de compasso com a verdadeira. O erro num instante isolado sobe e desce
# (é pequeno quando as curvas se cruzam), por isso medimos o **maior** erro em cada janela de 10 s. Ele cresce com o tempo de simulação.

# %% [markdown]
# ## 7. O que fazer a respeito?
#
# Esse problema, chamado de acúmulo de erros (compounding errors), aparece sempre que um modelo é usado em cadeia. Algumas saídas usadas
# na prática e na pesquisa:
#
# - **treinar em várias etapas:** em vez de medir o erro de um passo, simular vários passos durante o treino e medir o erro no fim, para a
#   rede aprender a não "desandar";
# - **embutir a Física:** redes construídas para conservar energia por construção (por exemplo, as redes hamiltonianas, Hamiltonian neural
#   networks) evitam a deriva da energia que vimos;
# - **corrigir com medições:** em previsão do tempo, os modelos são "puxados" de volta pelas observações reais a cada poucas horas;
# - **modelos híbridos:** a Física faz o grosso do trabalho, e a rede só corrige o que a Física não descreve (atrito, efeitos desconhecidos).
#
# A mesma lição vale para o Módulo 10: um modelo de linguagem também gera um token de cada vez a partir do que ele mesmo escreveu, e um
# deslize no começo pode levar o texto para longe.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Calcule à mão, pela fórmula de ângulos pequenos, o período de pêndulos com $L = 0{,}25$ m e $L = 4$ m. Depois meça o
# período na simulação para $L = 1$ m, soltando de 0,1 rad: use `simular(0.1, 0.0, 20, 0.001)` e meça o tempo entre as passagens de $\theta$
# de negativo para positivo. Bate com 2,007 s?
#
# **Exercício 2 (médio).** Meça o período (com o mesmo método) para amplitudes de 0,1 a 2,5 rad e faça um gráfico de período contra amplitude,
# com uma linha horizontal no valor da fórmula de ângulos pequenos. A partir de que amplitude a fórmula erra mais de 5%?
#
# **Exercício 3 (desafio).** Rode `trajetoria_rede` partindo de $\theta = 2{,}9$ rad (quase de cabeça para baixo), com $\omega = 0$, por 200 passos,
# e compare com a referência. A trajetória verdadeira sai da faixa de treino (que ia até 2,5 rad)? O que acontece com a rede? Compare o erro
# após 5 s com o do pêndulo solto de 1 rad.

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | pendulum | pêndulo | *A simple pendulum swings back and forth.* |
# | period | período | *The period of a one-meter pendulum is about two seconds.* |
# | amplitude | amplitude | *Large amplitudes make the period longer.* |
# | angular velocity | velocidade angular | *The angular velocity is zero at the turning points.* |
# | state | estado | *The state is the angle and the angular velocity.* |
# | energy conservation | conservação de energia | *Without friction, energy conservation holds.* |
# | semi-implicit Euler | Euler semi-implícito | *Semi-implicit Euler keeps the energy bounded.* |
# | symplectic | simplético | *Symplectic methods are used to simulate planets.* |
# | dynamics | dinâmica | *The network learns the dynamics of the pendulum.* |
# | rollout | simulação autorregressiva | *During a rollout, each prediction feeds the next step.* |
# | compounding errors | acúmulo de erros | *Compounding errors grow with every step.* |
# | time step | passo de tempo | *The network learned a time step of 0.05 seconds.* |
# | chaos | caos | *In chaos, tiny errors grow exponentially.* |
# | drift | deriva | *The energy of the network's pendulum drifts over time.* |

# %% [markdown]
# ## Resumo
#
# - O pêndulo obedece a $\ddot\theta = -(g/L)\sin\theta$; para ângulos pequenos, o período é $2\pi\sqrt{L/g}$.
# - O Euler simples faz a energia crescer; o semi-implícito, uma troca de ordem, mantém a energia sob controle.
# - Uma rede aprende muito bem um passo da dinâmica (prevendo a variação do estado).
# - Usada em cadeia, por 400 passos, a rede acumula erros: a fase escorrega e a energia deriva.
# - Treino em várias etapas, Física embutida, correção com medições e modelos híbridos ajudam a conter o acúmulo de erros.
#
# **Próximo passo:** os outros bônus, redes convolucionais para imagens (B2) e um projeto seu (B3), se você quiser continuar.

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
# $L = 0{,}25$ m: $T = 2\pi\sqrt{0{,}25/9{,}8} \approx 2\pi \cdot 0{,}1597 \approx 1{,}003$ s.
# $L = 4$ m: $T = 2\pi\sqrt{4/9{,}8} \approx 2\pi \cdot 0{,}6389 \approx 4{,}014$ s. (Quatro vezes o comprimento, o dobro do período.)

# %%
def medir_periodo(t, theta):
    """Período médio: tempo entre passagens de θ de negativo para positivo (com interpolação linear)."""
    cruzamentos = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0]
    instantes = t[cruzamentos] - theta[cruzamentos] * (t[cruzamentos + 1] - t[cruzamentos]) / (theta[cruzamentos + 1] - theta[cruzamentos])
    return np.mean(np.diff(instantes))


t_ex1, theta_ex1, _ = simular(0.1, 0.0, 20, 0.001)
print(f"período simulado: {medir_periodo(t_ex1, theta_ex1):.4f} s   fórmula: {2 * np.pi * np.sqrt(L / g):.4f} s")

# %% [markdown]
# ### Solução do Exercício 2

# %%
amplitudes = np.linspace(0.1, 2.5, 25)
periodos = []
for amplitude in amplitudes:
    t_a, theta_a, _ = simular(amplitude, 0.0, 30, 0.001)
    periodos.append(medir_periodo(t_a, theta_a))
periodos = np.array(periodos)
periodo_formula = 2 * np.pi * np.sqrt(L / g)

plt.figure(figsize=(7, 4))
plt.plot(amplitudes, periodos, marker="o", label="simulação")
plt.axhline(periodo_formula, color="gray", linestyle="--", label="fórmula de ângulos pequenos")
plt.xlabel("amplitude (rad)")
plt.ylabel("período (s)")
plt.legend()
plt.grid(True)
plt.show()

primeira = amplitudes[np.argmax(periodos > 1.05 * periodo_formula)]
print(f"a fórmula erra mais de 5% a partir de cerca de {primeira:.1f} rad ({np.degrees(primeira):.0f}°)")

# %% [markdown]
# O período cresce com a amplitude: perto do topo, o pêndulo "demora" para virar. A fórmula de ângulos pequenos vale bem até uns
# 30° e passa a errar mais de 5% por volta de 50°.

# %% [markdown]
# ### Solução do Exercício 3

# %%
verdade_alta = trajetoria_verdadeira(2.9, 0.0, 200)
previsao_alta = trajetoria_rede(2.9, 0.0, 200)
tempos_alta = np.arange(201) * DT_REDE
print(f"maior |θ| na trajetória verdadeira: {np.abs(verdade_alta[:, 0]).max():.2f} rad (o treino ia até 2,5)")
print(f"erro em θ após 5 s: começo em 2,9 rad {abs(previsao_alta[100, 0] - verdade_alta[100, 0]):.3f} rad   "
      f"começo em 1 rad {abs(previsao[100, 0] - verdade[100, 0]):.3f} rad")

plt.figure(figsize=(8, 4))
plt.plot(tempos_alta, verdade_alta[:, 0], label="Física (referência)")
plt.plot(tempos_alta, previsao_alta[:, 0], "--", label="rede")
plt.axhspan(-2.5, 2.5, color="green", alpha=0.08, label="faixa de θ do treino")
plt.xlabel("tempo (s)")
plt.ylabel("θ (rad)")
plt.title("Solto de 2,9 rad")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# A trajetória verdadeira passa boa parte do tempo acima de 2,5 rad, fora do que a rede viu. Ali a rede extrapola (Módulo 06) e erra bem mais
# por passo; somado ao acúmulo de erros, o resultado se afasta muito mais depressa do que no caso de 1 rad. Os dois problemas se reforçam:
# um erro grande pode empurrar a simulação para regiões ainda menos conhecidas, onde a rede erra mais.
