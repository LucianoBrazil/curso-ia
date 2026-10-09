# %% [markdown]
# # Módulo 01 — O que é "aprender": ajustando uma reta
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/01-ajustando-uma-reta.ipynb)
#
# **Objetivo.** Entender, sem mágica, o que significa uma máquina "aprender".
# Você vai medir o quanto um modelo erra e depois corrigir o modelo, passo a passo, na direção que diminui o erro.
#
# **O que você vai construir.** Um programa que descobre a velocidade de um carro a partir de medições com ruído,
# usando o gradiente descendente (gradient descent), o mesmo método que treina as grandes redes neurais.

# %% [markdown]
# ## 1. A intuição: o carro e o radar
#
# Um carro anda numa estrada reta com velocidade constante. Um fiscal anota, a cada poucos décimos de segundo,
# a que distância o carro está de uma marca no chão. As anotações não são perfeitas: há ruído (noise).
#
# A Física diz que, com velocidade constante $v$, a distância cresce em linha reta com o tempo:
#
# $$ d = v \cdot t $$
#
# O problema é o contrário do de costume: **conhecemos as medições, mas não conhecemos $v$**.
# Queremos o valor de $v$ que faz a reta passar o mais perto possível de todos os pontos.
#
# Em IA, chamamos a fórmula $d = w \cdot t$ de modelo (model). O número $w$, que o computador vai ajustar,
# é um parâmetro (parameter), também chamado de peso (weight). Aqui, o peso é a própria velocidade.
#
# "Aprender" vai significar isto: **ajustar $w$ até o modelo errar o mínimo possível.**

# %% [markdown]
# ## 2. Os dados
#
# Vamos simular o experimento. O carro anda a 20 m/s (72 km/h) e passa pela marca no instante zero.
# Medimos a distância em 25 instantes entre 0 e 10 s. Cada medição tem um ruído de alguns metros.
#
# Fixamos a semente (seed) para os resultados se repetirem, como no Módulo 00.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

velocidade_real = 20.0                       # m/s; o computador NÃO vai olhar este número
tempo = np.linspace(0, 10, 25)               # instantes das medições (s)
ruido = rng.normal(0, 10, size=tempo.shape)  # erro de medição (m)
distancia = velocidade_real * tempo + ruido  # o que o fiscal anotou (m)

plt.figure(figsize=(7, 4))
plt.scatter(tempo, distancia, color="tab:red", label="medições")
plt.xlabel("tempo (s)")
plt.ylabel("distância (m)")
plt.title("Carro em velocidade constante, medições com ruído")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Os pontos formam quase uma reta. Mas qual reta? Vamos testar alguns chutes.
#
# A função `prever` é o nosso modelo. O parâmetro `b` (posição inicial) fica em zero por enquanto;
# ele vai ser útil na seção 8.

# %%
def prever(tempo, w, b=0.0):
    """Modelo linear: distância prevista = w * tempo + b."""
    return w * tempo + b


plt.figure(figsize=(7, 4))
plt.scatter(tempo, distancia, color="tab:red", label="medições")
for w_chute in [10, 20, 30]:
    plt.plot(tempo, prever(tempo, w_chute), label=f"chute w = {w_chute} m/s")
plt.xlabel("tempo (s)")
plt.ylabel("distância (m)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# A olho, o chute de 20 m/s parece o melhor. Mas "a olho" não serve para um computador.
# Precisamos de um número que diga **o quanto** cada chute erra.

# %% [markdown]
# ## 3. Medindo o erro: o erro quadrático médio
#
# Para cada medição $i$, o resíduo (residual) é a diferença entre o que o modelo prevê e o que foi medido:
#
# $$ e_i = w \, t_i - d_i $$
#
# Elevamos cada resíduo ao quadrado (assim erros para cima e para baixo não se cancelam, e erros grandes pesam mais)
# e tiramos a média. Isso é o erro quadrático médio (mean squared error, MSE), que já apareceu no Módulo 00:
#
# $$ E(w) = \frac{1}{n} \sum_{i=1}^{n} \big( w\, t_i - d_i \big)^2 $$
#
# Em IA, a função que mede o erro se chama função de perda (loss function). Treinar um modelo é procurar o mínimo dela.
#
# **Exemplo à mão.** Três medições: $t = 1, 2, 3$ s e $d = 21, 39, 62$ m. Chute: $w = 20$.
#
# | $t$ | $d$ medido | previsto $20t$ | resíduo $e$ | $e^2$ |
# |---|---|---|---|---|
# | 1 | 21 | 20 | −1 | 1 |
# | 2 | 39 | 40 | +1 | 1 |
# | 3 | 62 | 60 | −2 | 4 |
#
# $$ E(20) = \frac{1 + 1 + 4}{3} = 2 $$

# %%
def erro_quadratico_medio(previsto, medido):
    """Média dos quadrados das diferenças (vimos isso no Módulo 00)."""
    return np.mean((previsto - medido) ** 2)


# Conferindo o exemplo feito à mão
tempo_mini = np.array([1.0, 2.0, 3.0])
dist_mini = np.array([21.0, 39.0, 62.0])
print(erro_quadratico_medio(prever(tempo_mini, 20), dist_mini))

# %% [markdown]
# Agora, o erro dos três chutes nos dados do carro:

# %%
for w_chute in [10, 20, 30]:
    erro = erro_quadratico_medio(prever(tempo, w_chute), distancia)
    print(f"w = {w_chute} m/s  ->  erro = {erro:8.1f} m²")

# %% [markdown]
# ## 4. A paisagem do erro
#
# Para cada valor possível de $w$ existe um erro $E(w)$. Se desenharmos todos, temos a "paisagem do erro"
# (loss landscape). Imagine um vale: a altura do terreno é o erro. Queremos chegar ao fundo do vale.

# %%
def erro_de_w(w):
    """Erro do modelo nos dados do carro, para um valor de w."""
    return erro_quadratico_medio(prever(tempo, w), distancia)


valores_w = np.linspace(0, 40, 200)
paisagem = np.array([erro_de_w(w) for w in valores_w])

plt.figure(figsize=(7, 4))
plt.plot(valores_w, paisagem)
plt.xlabel("w (m/s)")
plt.ylabel("erro E(w) (m²)")
plt.title("A paisagem do erro: uma parábola")
plt.grid(True)
plt.show()

# %% [markdown]
# A curva é uma parábola com a "boca" para cima. O fundo fica perto de 20 m/s.
#
# Aqui temos só um parâmetro, então poderíamos simplesmente testar 200 valores, como no Módulo 00.
# Mas uma rede neural tem milhares ou bilhões de parâmetros. Testar todas as combinações é impossível.
# Precisamos de um jeito de **descer o vale** sem conhecer o mapa inteiro. Para isso, basta saber a inclinação do chão
# onde estamos pisando.

# %% [markdown]
# ## 5. A derivada: a inclinação do terreno
#
# Imagine que você está de olhos vendados numa encosta. Você não vê o fundo do vale, mas sente com os pés
# se o chão desce para a esquerda ou para a direita. Essa inclinação é a derivada (derivative).
#
# **Inclinação de uma reta** você já conhece do ensino médio: quanto $y$ sobe quando $x$ anda 1.
# Uma curva não é uma reta, mas se dermos um zoom enorme num ponto, ela parece reta. A inclinação dessa
# "reta do zoom" (a reta tangente) é a derivada naquele ponto. Na prática, damos um passinho $h$ para cada lado:
#
# $$ E'(w) \approx \frac{E(w + h) - E(w - h)}{2h} \qquad (h \text{ bem pequeno}) $$
#
# **Uma fórmula exata.** Olhe um único termo da soma, $(w\,t - d)^2$, e troque $w$ por $w + h$.
# Pelo quadrado da soma, $(A + B)^2 = A^2 + 2AB + B^2$, com $A = w\,t - d$ e $B = h\,t$:
#
# $$ \big((w + h)\,t - d\big)^2 = (w\,t - d)^2 + 2\,(w\,t - d)\,t\,h + t^2 h^2 $$
#
# A variação do termo é $2\,(w\,t - d)\,t\,h + t^2 h^2$. Dividindo pelo passo $h$, sobra $2\,(w\,t - d)\,t + t^2 h$.
# Quando $h$ fica minúsculo, o último pedaço some. Fazendo a média de todos os termos:
#
# $$ E'(w) = \frac{2}{n} \sum_{i=1}^{n} \big( w\, t_i - d_i \big)\, t_i = \frac{2}{n} \sum_{i=1}^{n} e_i \, t_i $$
#
# Em palavras: some cada resíduo vezes o seu tempo, multiplique por 2 e divida por $n$.
#
# **Exemplo à mão**, com os três pontos da seção 3 e $w = 20$. Os resíduos eram $-1, +1, -2$:
#
# $$ E'(20) = \frac{2}{3}\big( (-1)(1) + (+1)(2) + (-2)(3) \big) = \frac{2}{3}(-5) \approx -3{,}33 $$
#
# A derivada é **negativa**: o erro diminui se aumentarmos $w$. O chão desce para a direita.

# %%
def derivada_erro(tempo, distancia, w):
    """Derivada do erro quadrático médio em relação a w (modelo d = w * t)."""
    residuo = prever(tempo, w) - distancia
    return 2 * np.mean(residuo * tempo)


# Conferindo a conta à mão
print(f"derivada no exemplo mini: {derivada_erro(tempo_mini, dist_mini, 20):.4f}")

# Conferindo a fórmula contra o "passinho para cada lado", nos dados do carro
h = 1e-4
for w in [10, 32]:
    numerica = (erro_de_w(w + h) - erro_de_w(w - h)) / (2 * h)
    print(f"w = {w}: fórmula = {derivada_erro(tempo, distancia, w):9.3f}   passinho = {numerica:9.3f}")

# %% [markdown]
# As duas contas batem. Vamos desenhar a reta tangente em dois pontos da paisagem.
# Em $w = 10$ a inclinação é negativa (desce para a direita); em $w = 32$ é positiva (desce para a esquerda).

# %%
plt.figure(figsize=(7, 4))
plt.plot(valores_w, paisagem, label="E(w)")
for w0 in [10, 32]:
    inclinacao = derivada_erro(tempo, distancia, w0)
    trecho = np.linspace(w0 - 6, w0 + 6, 10)
    tangente = erro_de_w(w0) + inclinacao * (trecho - w0)
    plt.plot(trecho, tangente, "--", linewidth=2, label=f"tangente em w = {w0} (inclinação {inclinacao:.0f})")
    plt.scatter([w0], [erro_de_w(w0)], color="black", zorder=3)
plt.xlabel("w (m/s)")
plt.ylabel("erro E(w) (m²)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. O gradiente descendente
#
# A regra é simples: **dê um passo no sentido contrário ao da inclinação.**
#
# $$ w_{\text{novo}} = w - \eta \cdot E'(w) $$
#
# - Se $E'(w) < 0$ (chão descendo para a direita), $w$ aumenta.
# - Se $E'(w) > 0$ (chão descendo para a esquerda), $w$ diminui.
# - Perto do fundo, a inclinação fica pequena e os passos encolhem sozinhos.
#
# O número $\eta$ (letra grega "eta") é a taxa de aprendizado (learning rate): o tamanho do passo.
# Cada repetição da regra é um passo, ou iteração (iteration).
# O nome "gradiente" vai fazer sentido na seção 8, quando houver dois parâmetros.
#
# **Exemplo à mão**, no exemplo mini, com $w = 20$ e $\eta = 0{,}01$:
#
# $$ w_{\text{novo}} = 20 - 0{,}01 \cdot (-3{,}33) = 20 + 0{,}0333 = 20{,}0333 $$
#
# Vamos repetir isso no computador, começando de um chute ruim: $w = 0$.

# %%
w = 0.0
taxa = 0.01
for passo in range(6):
    inclinacao = derivada_erro(tempo, distancia, w)
    print(f"passo {passo}: w = {w:7.3f}   erro = {erro_de_w(w):8.1f}   inclinação = {inclinacao:9.2f}")
    w = w - taxa * inclinacao

# %% [markdown]
# Em poucos passos, $w$ saiu de 0 e chegou perto de 20. Vamos empacotar o processo numa função
# que guarda o histórico, para podermos desenhar o aprendizado.

# %%
def gradiente_descendente_1d(tempo, distancia, w_inicial, taxa, passos):
    """Ajusta w no modelo d = w * t. Devolve o histórico de w e do erro."""
    w = w_inicial
    historico_w = [w]
    historico_erro = [erro_quadratico_medio(prever(tempo, w), distancia)]
    for passo in range(passos):
        w = w - taxa * derivada_erro(tempo, distancia, w)
        historico_w.append(w)
        historico_erro.append(erro_quadratico_medio(prever(tempo, w), distancia))
    return np.array(historico_w), np.array(historico_erro)


historico_w, historico_erro = gradiente_descendente_1d(tempo, distancia, w_inicial=0.0, taxa=0.01, passos=30)
print(f"w final = {historico_w[-1]:.3f} m/s")

# %% [markdown]
# **O aprendizado acontecendo.** À esquerda, o erro caindo a cada passo (em escala logarítmica, onde cada
# marca do eixo vale 10 vezes a anterior). À direita, a reta se ajustando aos pontos.

# %%
fig, (eixo_erro, eixo_reta) = plt.subplots(1, 2, figsize=(12, 4))

eixo_erro.plot(historico_erro, marker="o")
eixo_erro.set_yscale("log")
eixo_erro.set_xlabel("passo")
eixo_erro.set_ylabel("erro (m²)")
eixo_erro.set_title("O erro caindo")
eixo_erro.grid(True)

eixo_reta.scatter(tempo, distancia, color="tab:red", label="medições")
for passo in [0, 1, 2, 4, 30]:
    eixo_reta.plot(tempo, prever(tempo, historico_w[passo]), label=f"passo {passo}: w = {historico_w[passo]:.1f}")
eixo_reta.set_xlabel("tempo (s)")
eixo_reta.set_ylabel("distância (m)")
eixo_reta.set_title("A reta se ajustando")
eixo_reta.legend()
eixo_reta.grid(True)
plt.show()

# %% [markdown]
# O erro não chega a zero, e nem deveria: as medições têm ruído. O modelo para no melhor compromisso possível.

# %% [markdown]
# ## 7. A taxa de aprendizado: nem muito pequena, nem muito grande
#
# A taxa $\eta$ é escolhida por nós, não pelo modelo. Valores assim, que ajustamos "por fora",
# se chamam hiperparâmetros (hyperparameters). Vamos comparar três escolhas, com 12 passos cada:
#
# - **pequena demais:** passos minúsculos, o aprendizado se arrasta;
# - **boa:** chega rápido ao fundo;
# - **grande demais:** cada passo pula por cima do fundo e cai **mais alto** do outro lado. O erro cresce: o método diverge (diverges).

# %%
taxas = {"pequena demais": 0.0005, "boa": 0.01, "grande demais": 0.032}

fig, eixos = plt.subplots(1, 3, figsize=(15, 4))
for eixo, (nome, taxa) in zip(eixos, taxas.items()):
    hist_w, hist_erro = gradiente_descendente_1d(tempo, distancia, w_inicial=0.0, taxa=taxa, passos=12)
    faixa = np.linspace(min(hist_w.min(), 0) - 5, max(hist_w.max(), 40) + 5, 300)
    eixo.plot(faixa, [erro_de_w(w) for w in faixa], color="lightgray")
    eixo.plot(hist_w, hist_erro, marker="o", color="tab:blue")
    eixo.set_title(f"taxa {taxa} ({nome})\nw final = {hist_w[-1]:.1f}")
    eixo.set_xlabel("w (m/s)")
    eixo.set_ylabel("erro (m²)")
    eixo.grid(True)
plt.show()

# %% [markdown]
# Na figura da direita, o "zigue-zague" se abre: $w$ pula de um lado para o outro do vale, cada vez mais longe.
#
# Existe um limite. Para esta paisagem, a taxa precisa ser menor que $1 / \overline{t^2}$, onde $\overline{t^2}$ é a média
# dos tempos ao quadrado. (O Exercício 2 explora isso.) Em problemas reais não há fórmula para o limite:
# testamos alguns valores e observamos a curva do erro.

# %%
print(f"limite teórico para a taxa: {1 / np.mean(tempo**2):.4f}")

# %% [markdown]
# ## 8. Dois parâmetros: o carro que já estava na estrada
#
# Num segundo experimento, o cronômetro começa quando o carro **já está** a uns 30 m da marca. Agora o modelo precisa
# de dois parâmetros: a velocidade $w$ e a posição inicial $b$ (em IA, $b$ se chama viés, ou bias):
#
# $$ d = w\, t + b $$
#
# O erro passa a depender dos dois: $E(w, b)$. A paisagem deixa de ser uma curva e vira uma superfície, como um terreno.
#
# Para descer, precisamos da inclinação em cada direção. A derivada parcial (partial derivative) é a inclinação
# quando mexemos em **um** parâmetro e seguramos o outro parado. Pela mesma conta da seção 5:
#
# $$ \frac{\partial E}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} e_i\, t_i \qquad\qquad \frac{\partial E}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i $$
#
# com $e_i = w\,t_i + b - d_i$. O símbolo $\partial$ (lê-se "dê redondo") indica derivada parcial.
# As duas derivadas juntas formam o gradiente (gradient): uma seta que aponta para onde o terreno **sobe** mais.
# O gradiente descendente anda no sentido contrário dessa seta, atualizando os dois parâmetros ao mesmo tempo:
#
# $$ w \leftarrow w - \eta \, \frac{\partial E}{\partial w} \qquad\qquad b \leftarrow b - \eta \, \frac{\partial E}{\partial b} $$

# %%
posicao_inicial_real = 30.0   # m; de novo, o computador não olha
distancia_b = velocidade_real * tempo + posicao_inicial_real + rng.normal(0, 10, size=tempo.shape)


def gradiente_descendente(tempo, distancia, w_inicial, b_inicial, taxa, passos):
    """Ajusta w e b no modelo d = w * t + b. Devolve os históricos de w, b e do erro."""
    w, b = w_inicial, b_inicial
    historico_w, historico_b, historico_erro = [w], [b], [erro_quadratico_medio(prever(tempo, w, b), distancia)]
    for passo in range(passos):
        residuo = prever(tempo, w, b) - distancia
        derivada_w = 2 * np.mean(residuo * tempo)
        derivada_b = 2 * np.mean(residuo)
        w = w - taxa * derivada_w
        b = b - taxa * derivada_b
        historico_w.append(w)
        historico_b.append(b)
        historico_erro.append(erro_quadratico_medio(prever(tempo, w, b), distancia))
    return np.array(historico_w), np.array(historico_b), np.array(historico_erro)


hist_w2, hist_b2, hist_erro2 = gradiente_descendente(tempo, distancia_b, 0.0, 0.0, taxa=0.02, passos=1000)
print(f"w final = {hist_w2[-1]:.3f} m/s   b final = {hist_b2[-1]:.3f} m")

# %% [markdown]
# Vamos ver o caminho percorrido no "mapa" da paisagem. Cada linha fechada liga pontos de mesmo erro,
# como as curvas de nível de um mapa de relevo. Para montar o mapa, calculamos o erro numa grade de valores de $w$ e $b$.
# A função `enumerate` entrega, a cada volta do laço, a posição e o valor do item.

# %%
grade_w = np.linspace(-2, 40, 120)
grade_b = np.linspace(-20, 70, 120)
mapa_erro = np.zeros((len(grade_b), len(grade_w)))
for i, b in enumerate(grade_b):
    for j, w in enumerate(grade_w):
        mapa_erro[i, j] = erro_quadratico_medio(prever(tempo, w, b), distancia_b)

niveis = np.logspace(np.log10(mapa_erro.min()), np.log10(mapa_erro.max()), 25)

fig, (eixo_mapa, eixo_erro) = plt.subplots(1, 2, figsize=(13, 4.5))
eixo_mapa.contour(grade_w, grade_b, mapa_erro, levels=niveis, cmap="viridis")
eixo_mapa.plot(hist_w2, hist_b2, color="tab:red", marker=".", markersize=3, label="caminho do gradiente descendente")
eixo_mapa.scatter([hist_w2[-1]], [hist_b2[-1]], color="black", zorder=3, label="final")
eixo_mapa.set_xlabel("w (m/s)")
eixo_mapa.set_ylabel("b (m)")
eixo_mapa.set_title("Mapa do erro E(w, b)")
eixo_mapa.legend(loc="lower right")

eixo_erro.plot(hist_erro2)
eixo_erro.set_yscale("log")
eixo_erro.set_xlabel("passo")
eixo_erro.set_ylabel("erro (m²)")
eixo_erro.set_title("O erro caindo, em duas fases")
eixo_erro.grid(True)
plt.show()

# %% [markdown]
# Repare em duas fases. Primeiro, o caminho desce rápido até o fundo de um vale comprido e estreito.
# Depois, escorrega devagar ao longo do vale até o mínimo: o $b$ demora centenas de passos para se acertar.
# Isso acontece porque $t$ vai até 10, enquanto o "atributo" de $b$ vale sempre 1: as duas direções têm escalas
# muito diferentes. No Módulo 02 você vai ver que normalizar os dados resolve isso.

# %% [markdown]
# ## 9. A solução exata: mínimos quadrados
#
# No fundo do vale, o chão é plano: a derivada vale zero. Para o modelo de um parâmetro, $d = w\,t$:
#
# $$ \frac{2}{n} \sum_{i} (w\, t_i - d_i)\, t_i = 0 \;\Rightarrow\; w \sum_i t_i^2 = \sum_i t_i\, d_i \;\Rightarrow\; w = \frac{\sum_i t_i\, d_i}{\sum_i t_i^2} $$
#
# **Exemplo à mão**, com os três pontos: $\sum t\,d = 21 + 78 + 186 = 285$ e $\sum t^2 = 1 + 4 + 9 = 14$, logo
# $w = 285 / 14 \approx 20{,}36$ m/s.
#
# Para o modelo $d = w\,t + b$, igualando as duas derivadas parciais a zero, chega-se a (com $\bar{t}$ e $\bar{d}$ as médias):
#
# $$ w = \frac{\sum_i (t_i - \bar{t})(d_i - \bar{d})}{\sum_i (t_i - \bar{t})^2} \qquad\qquad b = \bar{d} - w\, \bar{t} $$
#
# Esse é o método dos mínimos quadrados (least squares). Vamos comparar com o gradiente descendente
# e com a função pronta `np.polyfit` do NumPy.

# %%
def minimos_quadrados(tempo, distancia):
    """Solução exata para d = w * t + b."""
    t_medio, d_medio = tempo.mean(), distancia.mean()
    w = np.sum((tempo - t_medio) * (distancia - d_medio)) / np.sum((tempo - t_medio) ** 2)
    b = d_medio - w * t_medio
    return w, b


print(f"exemplo mini (só w): {np.sum(tempo_mini * dist_mini) / np.sum(tempo_mini**2):.4f} m/s")

w_exato, b_exato = minimos_quadrados(tempo, distancia_b)
w_polyfit, b_polyfit = np.polyfit(tempo, distancia_b, 1)
print(f"gradiente descendente: w = {hist_w2[-1]:.4f}   b = {hist_b2[-1]:.4f}")
print(f"mínimos quadrados:     w = {w_exato:.4f}   b = {b_exato:.4f}")
print(f"np.polyfit:            w = {w_polyfit:.4f}   b = {b_polyfit:.4f}")
print(f"valores reais:         w = {velocidade_real:.4f}   b = {posicao_inicial_real:.4f}")

# %% [markdown]
# Os três métodos concordam. Nenhum acerta exatamente 20 e 30, por causa do ruído: com mais medições, chegariam mais perto.
#
# **Então por que usar o gradiente descendente, se existe fórmula?** Porque a fórmula só existe para modelos simples
# como este. Para uma rede neural não há fórmula nenhuma. Mas sempre dá para calcular a inclinação e dar um passo.
# O gradiente descendente é o motor de quase todo o aprendizado de máquina (machine learning).

# %%
plt.figure(figsize=(7, 4))
plt.scatter(tempo, distancia_b, color="tab:red", label="medições")
plt.plot(tempo, prever(tempo, w_exato, b_exato), label=f"ajuste: d = {w_exato:.1f} t + {b_exato:.1f}")
plt.xlabel("tempo (s)")
plt.ylabel("distância (m)")
plt.title("Reta ajustada ao segundo experimento")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Com os três pontos do exemplo mini (`tempo_mini`, `dist_mini`), calcule o erro para
# $w = 19$, $w = 20$, $w = 21$ e $w = 285/14$. Qual é o menor? Calcule também a derivada em $w = 21$:
# qual é o sinal, e para que lado o gradiente descendente moveria $w$?
#
# **Exercício 2 (médio).** Rode `gradiente_descendente_1d` nos dados do carro, com 100 passos, para as taxas
# 0,01; 0,02; 0,028; 0,029; 0,030 e 0,032. Para cada uma, mostre o $w$ final e o erro final.
# Onde fica a fronteira entre convergir (converge) e divergir? Compare com o limite teórico $1/\overline{t^2}$.
#
# **Exercício 3 (desafio).** Volte à queda livre do Módulo 00, onde $s = \frac{1}{2}\, g\, t^2$.
# Gere 20 medições com `rng.normal(0, 1.5, ...)` para $t$ entre 0,2 e 3 s (semente 7). Escreva a derivada do erro em relação a $g$
# (dica: é a mesma conta da seção 5, trocando $t$ por $\frac{1}{2}t^2$) e use o gradiente descendente para descobrir $g$,
# começando de $g = 0$. Compare com a solução exata $g = \sum x_i s_i / \sum x_i^2$, onde $x_i = \frac{1}{2} t_i^2$.

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | model | modelo | *Our model is a straight line.* |
# | parameter / weight | parâmetro / peso | *The weight w is the speed of the car.* |
# | bias | viés | *The bias b shifts the line up or down.* |
# | residual | resíduo | *A residual is the prediction minus the measurement.* |
# | loss function | função de perda | *The loss function tells us how wrong the model is.* |
# | mean squared error | erro quadrático médio | *We minimize the mean squared error.* |
# | loss landscape | paisagem do erro | *The loss landscape looks like a valley.* |
# | slope | inclinação | *The slope is negative, so we move to the right.* |
# | derivative | derivada | *The derivative is zero at the bottom.* |
# | gradient | gradiente | *The gradient points uphill.* |
# | gradient descent | gradiente descendente | *Gradient descent walks downhill step by step.* |
# | learning rate | taxa de aprendizado | *A large learning rate can make training diverge.* |
# | step / iteration | passo / iteração | *After thirty iterations, the error stopped falling.* |
# | converge / diverge | convergir / divergir | *With a good learning rate, the method converges.* |
# | least squares | mínimos quadrados | *Least squares gives the exact answer for a line.* |

# %% [markdown]
# ## Resumo
#
# - "Aprender" é ajustar os parâmetros de um modelo para diminuir uma função de perda, como o erro quadrático médio.
# - A paisagem do erro mostra o erro para cada valor dos parâmetros; o melhor modelo está no fundo do vale.
# - A derivada é a inclinação do terreno; o gradiente reúne as derivadas parciais de todos os parâmetros.
# - O gradiente descendente dá passos contra a inclinação; a taxa de aprendizado não pode ser nem pequena nem grande demais.
# - Para uma reta existe solução exata (mínimos quadrados); o gradiente descendente vale também quando não há fórmula.
#
# **Próximo módulo:** muitas variáveis ao mesmo tempo, com vetores e matrizes, e por que normalizar os dados acelera o aprendizado.

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

# %%
for w in [19, 20, 21, 285 / 14]:
    print(f"w = {w:7.4f}  ->  erro = {erro_quadratico_medio(prever(tempo_mini, w), dist_mini):.4f}")

print(f"derivada em w = 21: {derivada_erro(tempo_mini, dist_mini, 21):.4f}")

# %% [markdown]
# O menor erro é o de $w = 285/14 \approx 20{,}36$, a solução exata. Em $w = 21$ a derivada é positiva
# (o terreno sobe para a direita), então o gradiente descendente diminuiria $w$.

# %% [markdown]
# ### Solução do Exercício 2

# %%
for taxa in [0.01, 0.02, 0.028, 0.029, 0.030, 0.032]:
    hist_w, hist_erro = gradiente_descendente_1d(tempo, distancia, w_inicial=0.0, taxa=taxa, passos=100)
    print(f"taxa {taxa:.3f}:  w final = {hist_w[-1]:12.3f}   erro final = {hist_erro[-1]:14.1f}")

print(f"limite teórico: {1 / np.mean(tempo**2):.4f}")

# %% [markdown]
# Até 0,028 o método converge bem. Com 0,029, bem perto do limite, ainda converge, mas devagar: depois de 100 passos
# ainda está oscilando em torno do fundo. A partir de 0,030 ele diverge.
# A fronteira bate com o limite teórico, cerca de 0,0294.
#
# De onde vem o limite? A derivada da derivada (a "curvatura" da parábola) vale $2\,\overline{t^2}$. A cada passo, a distância
# até o fundo é multiplicada por $1 - \eta \cdot 2\,\overline{t^2}$. Se esse número passar de $-1$, a distância cresce em vez de diminuir.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def posicao_queda(t, g=9.8):
    """Vimos isso no Módulo 00: distância percorrida em queda livre."""
    return 0.5 * g * t**2


rng7 = np.random.default_rng(7)
t_queda = np.linspace(0.2, 3.0, 20)
s_queda = posicao_queda(t_queda) + rng7.normal(0, 1.5, size=t_queda.shape)

x = 0.5 * t_queda**2   # o "atributo" que multiplica g
g = 0.0
taxa_g = 0.05
for passo in range(200):
    residuo = g * x - s_queda
    derivada_g = 2 * np.mean(residuo * x)
    g = g - taxa_g * derivada_g

g_exato = np.sum(x * s_queda) / np.sum(x**2)
print(f"gradiente descendente: g = {g:.4f} m/s²")
print(f"solução exata:         g = {g_exato:.4f} m/s²")
