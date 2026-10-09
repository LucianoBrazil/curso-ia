# %% [markdown]
# # Módulo 03 — Classificação: regressão logística
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/03-regressao-logistica.ipynb)
#
# **Objetivo.** Até aqui, os modelos previam um número (distância, consumo). Agora vamos prever uma resposta de
# "sim ou não": a classificação (classification). Você vai ver como transformar um número em probabilidade e
# como medir o erro de uma probabilidade.
#
# **O que você vai construir.** Um modelo que estima a chance de um aluno ser aprovado numa prova a partir das horas de
# estudo e dos exercícios resolvidos, desenha a fronteira entre "aprovado" e "reprovado" e avalia os próprios acertos.

# %% [markdown]
# ## 1. A intuição: de "quanto" para "qual"
#
# Uma escola quer prever quem vai passar na prova final. Para cada aluno, sabe duas coisas: quantas horas estudou
# e quantos exercícios resolveu. A resposta é 1 (aprovado) ou 0 (reprovado). Quando há só duas respostas
# possíveis, falamos em classificação binária (binary classification). Cada resposta possível é uma classe (class).
#
# Um bom modelo não deve responder só "passa" ou "não passa". Deve dar uma **probabilidade** (probability):
# "este aluno tem 80% de chance de passar".
#
# **Probabilidade em uma frase.** É um número entre 0 e 1 que diz com que frequência algo acontece. Se, entre 100 alunos
# com o mesmo perfil, 80 passaram, a probabilidade de passar é 0,8. A de reprovar é o que falta: $1 - 0{,}8 = 0{,}2$.

# %% [markdown]
# ## 2. Os dados: 200 alunos
#
# Vamos simular uma turma. Por trás dos dados existe uma regra escondida (que o modelo **não** vai ver):
# mais horas e mais exercícios aumentam a chance de passar. Mas há sorte envolvida: dois alunos iguais podem ter
# resultados diferentes. Por isso sorteamos o resultado de cada aluno a partir da sua probabilidade.

# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n_alunos = 200

horas_estudo = rng.uniform(0, 20, n_alunos)
exercicios = rng.integers(0, 51, n_alunos).astype(float)

# Regra escondida (o modelo não olha): chance de passar
pontuacao_real = 0.4 * horas_estudo + 0.1 * exercicios - 7
chance_real = 1 / (1 + np.exp(-pontuacao_real))
aprovado = (rng.uniform(0, 1, n_alunos) < chance_real).astype(float)   # 1 = aprovado, 0 = reprovado

X = np.column_stack([horas_estudo, exercicios])
y = aprovado
print("forma de X:", X.shape)
print(f"aprovados: {int(y.sum())} de {n_alunos}")

# %% [markdown]
# A linha `rng.uniform(0, 1) < chance` é um sorteio: um número aleatório entre 0 e 1 fica abaixo de 0,8 em 80% das vezes.
# O `astype(float)` transforma `True`/`False` em 1,0/0,0.
#
# No gráfico, cada ponto é um aluno. Os aprovados ficam, em geral, no canto de cima à direita.

# %%
def desenhar_alunos(eixo, X, y):
    """Pontos coloridos por classe: aprovados em azul, reprovados em vermelho."""
    eixo.scatter(X[y == 1, 0], X[y == 1, 1], color="tab:blue", marker="o", s=20, label="aprovado")
    eixo.scatter(X[y == 0, 0], X[y == 0, 1], color="tab:red", marker="x", s=20, label="reprovado")
    eixo.set_xlabel("horas de estudo")
    eixo.set_ylabel("exercícios resolvidos")


fig, eixo = plt.subplots(figsize=(7, 5))
desenhar_alunos(eixo, X, y)
eixo.legend()
eixo.grid(True)
plt.show()

# %% [markdown]
# ## 3. Por que uma reta não serve
#
# Olhe só as horas de estudo. Se ajustarmos uma reta aos valores 0 e 1, como no Módulo 01, a reta passa de 1 para quem
# estudou muito e fica negativa para quem estudou pouco. Uma "probabilidade" de 1,3 ou de −0,2 não faz sentido.

# %%
inclinacao, intercepto = np.polyfit(horas_estudo, y, 1)
horas_grafico = np.linspace(-2, 24, 100)

plt.figure(figsize=(7, 4))
plt.scatter(horas_estudo, y, alpha=0.4, label="alunos (1 = aprovado)")
plt.plot(horas_grafico, inclinacao * horas_grafico + intercepto, color="tab:orange", label="reta ajustada")
plt.axhline(0, color="gray", linestyle=":")
plt.axhline(1, color="gray", linestyle=":")
plt.xlabel("horas de estudo")
plt.ylabel("aprovado?")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Precisamos de uma função que pegue **qualquer** número e o "espremer" entre 0 e 1.

# %% [markdown]
# ## 4. A função sigmoide
#
# A função sigmoide (sigmoid) faz exatamente isso:
#
# $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
#
# O número $e \approx 2{,}718$ é a base dos logaritmos naturais. O que importa é o comportamento de $e^{-z}$:
#
# - se $z$ é grande e positivo, $e^{-z}$ é quase zero, e $\sigma(z) \approx \frac{1}{1} = 1$;
# - se $z$ é grande e negativo, $e^{-z}$ é enorme, e $\sigma(z) \approx 0$;
# - se $z = 0$, $e^{0} = 1$, e $\sigma(0) = \frac{1}{2}$.
#
# **Exemplo à mão.** Com $e^{-2} \approx 0{,}135$:
#
# $$ \sigma(2) = \frac{1}{1 + 0{,}135} \approx 0{,}881 \qquad\qquad \sigma(-2) = \frac{1}{1 + e^{2}} = \frac{1}{1 + 7{,}389} \approx 0{,}119 $$
#
# Repare: $\sigma(-2) = 1 - \sigma(2)$. A curva é simétrica em torno do ponto $(0;\ 0{,}5)$.
#
# **Curiosidade de Física.** A distribuição de Fermi-Dirac, que diz a chance de um elétron ocupar um nível de energia $E$,
# é $f(E) = 1 / \big(e^{(E-\mu)/kT} + 1\big)$: é uma sigmoide, virada ao contrário.

# %%
def sigmoide(z):
    """Espreme qualquer número real para o intervalo entre 0 e 1."""
    return 1 / (1 + np.exp(-z))


print(f"sigmoide(0)  = {sigmoide(0):.3f}")
print(f"sigmoide(2)  = {sigmoide(2):.3f}")
print(f"sigmoide(-2) = {sigmoide(-2):.3f}")

valores_z = np.linspace(-8, 8, 200)
plt.figure(figsize=(7, 3.5))
plt.plot(valores_z, sigmoide(valores_z))
plt.axhline(0.5, color="gray", linestyle=":")
plt.axvline(0, color="gray", linestyle=":")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.title("A sigmoide")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 5. O modelo: regressão logística
#
# A ideia é juntar o que já sabemos com a sigmoide. Primeiro uma pontuação linear, como no Módulo 02; depois a sigmoide:
#
# $$ z = w_1 x_1 + w_2 x_2 + b = Xw + b \qquad\qquad p = \sigma(z) $$
#
# O número $p$ é a probabilidade, prevista pelo modelo, de o aluno ser aprovado. Esse modelo se chama
# regressão logística (logistic regression). O nome engana: apesar de "regressão", ele é usado para classificar.
#
# **Exemplo à mão**, com $w = (0{,}4;\ 0{,}1)$ e $b = -7$:
#
# - aluno com 15 h e 20 exercícios: $z = 0{,}4 \cdot 15 + 0{,}1 \cdot 20 - 7 = 6 + 2 - 7 = 1$, e $p = \sigma(1) \approx 0{,}73$;
# - aluno com 10 h e 30 exercícios: $z = 4 + 3 - 7 = 0$, e $p = 0{,}5$. Ele está exatamente "em cima do muro".

# %%
def prever_probabilidade(X, w, b):
    """Regressão logística: probabilidade da classe 1 para cada linha de X."""
    return sigmoide(X @ w + b)


alunos_exemplo = np.array([[15.0, 20.0],
                           [10.0, 30.0]])
print(prever_probabilidade(alunos_exemplo, np.array([0.4, 0.1]), -7.0))

# %% [markdown]
# ## 6. Medindo o erro de uma probabilidade: a entropia cruzada
#
# Pense na "surpresa" do modelo. Se ele diz 99% de chance de aprovação e o aluno é aprovado, quase não há surpresa.
# Se ele diz 99% e o aluno reprova, a surpresa é enorme, e o modelo merece uma punição enorme.
#
# A medida usada é a entropia cruzada (cross-entropy), também chamada de log loss. Para um aluno:
#
# $$ L = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big] $$
#
# Só uma das duas parcelas sobra, porque $y$ é 0 ou 1:
#
# - se o aluno passou ($y = 1$): $L = -\log p$;
# - se reprovou ($y = 0$): $L = -\log(1 - p)$.
#
# Aqui $\log$ é o logaritmo natural (ln). Basta saber: $\log 1 = 0$, e o logaritmo de um número próximo de zero é muito negativo.
# Logo $-\log p$ é zero quando o modelo acerta com certeza e cresce sem limite quando ele erra com certeza.
#
# **Exemplo à mão**, para um aluno aprovado ($y = 1$):
#
# | $p$ previsto | $L = -\log p$ |
# |---|---|
# | 0,9 | 0,105 |
# | 0,5 | 0,693 |
# | 0,1 | 2,303 |
# | 0,01 | 4,605 |
#
# A perda (loss) da turma inteira é a média das perdas de cada aluno.

# %%
def entropia_cruzada(p, y):
    """Entropia cruzada média. O clip evita log(0) quando p fica exatamente 0 ou 1."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


for p in [0.9, 0.5, 0.1, 0.01]:
    print(f"p = {p:4}: perda = {entropia_cruzada(np.array([p]), np.array([1.0])):.3f}")

# %% [markdown]
# ### Por que não usar o erro quadrático?
#
# Poderíamos usar $(p - y)^2$, como antes. O problema aparece quando o modelo está **muito errado**.
# Imagine um aluno aprovado ($y = 1$) para quem o modelo calculou $z = -6$, ou seja, $p = \sigma(-6) \approx 0{,}0025$.
#
# O gradiente descendente precisa da inclinação da perda em relação a $z$. Vamos calcular as duas (a fórmula da entropia
# cruzada é deduzida na seção 7):
#
# - erro quadrático: inclinação $= 2\,(p - y)\, p\,(1 - p) \approx 2 \cdot (-1) \cdot 0{,}0025 \cdot 1 \approx -0{,}005$;
# - entropia cruzada: inclinação $= p - y \approx -1$.
#
# Com o erro quadrático, a inclinação é quase zero justamente quando o modelo está mais errado: o terreno fica plano e o
# aprendizado empaca. A entropia cruzada continua "empurrando" com força.

# %%
z_errado = -6.0
p_errado = sigmoide(z_errado)
print(f"p = {p_errado:.4f}")
print(f"inclinação com erro quadrático:  {2 * (p_errado - 1) * p_errado * (1 - p_errado):.5f}")
print(f"inclinação com entropia cruzada: {p_errado - 1:.5f}")

valores_p = np.linspace(0.001, 0.999, 300)
plt.figure(figsize=(7, 4))
plt.plot(valores_p, (1 - valores_p) ** 2, label="erro quadrático (1 − p)²")
plt.plot(valores_p, -np.log(valores_p), label="entropia cruzada −log p")
plt.xlabel("probabilidade prevista p (o aluno foi aprovado, y = 1)")
plt.ylabel("perda")
plt.ylim(0, 5)
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# O erro quadrático nunca passa de 1, mesmo para um erro grosseiro. A entropia cruzada dispara quando $p$ vai para 0.

# %% [markdown]
# ## 7. O gradiente: um resultado surpreendentemente simples
#
# Para descer a perda, precisamos de $\frac{\partial L}{\partial w_j}$. O caminho tem três elos:
# $w_j$ muda $z$, $z$ muda $p$, e $p$ muda $L$. Cada elo tem a sua inclinação, e a inclinação total é o **produto** delas
# (é a regra da cadeia, ou chain rule, que o Módulo 04 vai explorar a fundo). Pense em engrenagens: se a primeira gira 2 vezes
# mais rápido que a segunda, e a segunda 3 vezes mais rápido que a terceira, a primeira gira 6 vezes mais rápido que a terceira.
#
# 1. **De $p$ para $L$:** derivando $L = -[y \log p + (1-y)\log(1-p)]$, usando que a derivada de $\log p$ é $1/p$:
#    $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p}$.
# 2. **De $z$ para $p$:** a sigmoide tem uma derivada elegante: $\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) = p\,(1 - p)$.
# 3. **De $w_j$ para $z$:** como $z = w_1 x_1 + w_2 x_2 + b$, a inclinação é simplesmente $x_j$.
#
# Multiplicando os dois primeiros elos, quase tudo se cancela:
#
# $$ \left( -\frac{y}{p} + \frac{1 - y}{1 - p} \right) p\,(1 - p) = -y\,(1 - p) + (1 - y)\,p = p - y $$
#
# Fazendo a média sobre os $n$ alunos:
#
# $$ \frac{\partial L}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)\, x_{ij} \qquad\qquad \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) $$
#
# É a mesma forma da regressão linear do Módulo 02 (sem o fator 2): resíduo vezes atributo. Em notação de matriz,
# $\nabla_w L = \frac{1}{n} X^T (p - y)$.
#
# Vamos conferir a derivada da sigmoide com o "passinho para cada lado" do Módulo 01.

# %%
h = 1e-5
for z in [-2.0, 0.0, 1.0]:
    numerica = (sigmoide(z + h) - sigmoide(z - h)) / (2 * h)
    formula = sigmoide(z) * (1 - sigmoide(z))
    print(f"z = {z:4}: passinho = {numerica:.6f}   fórmula p(1 − p) = {formula:.6f}")

# %%
def gradientes_logistica(X, y, w, b):
    """Gradiente da entropia cruzada média em relação a w e a b."""
    n = len(y)
    residuo = prever_probabilidade(X, w, b) - y
    return (X.T @ residuo) / n, np.sum(residuo) / n


def acuracia(classe_prevista, y):
    """Fração de acertos."""
    return np.mean(classe_prevista == y)


def treinar_logistica(X, y, taxa, passos, w_inicial=None, b_inicial=0.0):
    """Gradiente descendente para regressão logística. Devolve w, b e os históricos de perda e acurácia."""
    w = np.zeros(X.shape[1]) if w_inicial is None else np.array(w_inicial, dtype=float)
    b = b_inicial
    historico_perda, historico_acuracia = [], []
    for passo in range(passos + 1):
        p = prever_probabilidade(X, w, b)
        historico_perda.append(entropia_cruzada(p, y))
        historico_acuracia.append(acuracia(p >= 0.5, y))
        if passo == passos:
            break
        gradiente_w, gradiente_b = gradientes_logistica(X, y, w, b)
        w = w - taxa * gradiente_w
        b = b - taxa * gradiente_b
    return w, b, np.array(historico_perda), np.array(historico_acuracia)

# %% [markdown]
# A função `acuracia` (accuracy) compara a classe prevista com a real. A classe prevista é 1 quando $p \ge 0{,}5$.
# Vamos detalhar isso na seção 9.

# %% [markdown]
# ## 8. Treinando
#
# Horas vão até 20 e exercícios até 50: escalas diferentes. Como vimos no Módulo 02, normalizamos antes de treinar.
# As funções abaixo foram copiadas do Módulo 02.

# %%
def normalizar(X):
    """Vimos isso no Módulo 02: padroniza cada coluna (média 0, desvio padrão 1)."""
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    return (X - media) / desvio, media, desvio


def desnormalizar_pesos(w_norm, b_norm, media, desvio):
    """Vimos isso no Módulo 02: converte os pesos para as unidades originais."""
    w = w_norm / desvio
    b = b_norm - np.sum(w_norm * media / desvio)
    return w, b


X_norm, media_X, desvio_X = normalizar(X)
w_norm, b_norm, historico_perda, historico_acuracia = treinar_logistica(X_norm, y, taxa=0.5, passos=1000)
w_final, b_final = desnormalizar_pesos(w_norm, b_norm, media_X, desvio_X)

print(f"perda final: {historico_perda[-1]:.3f}   acurácia final: {historico_acuracia[-1]:.1%}")
print(f"pesos aprendidos: horas = {w_final[0]:.3f}, exercícios = {w_final[1]:.3f}, viés = {b_final:.3f}")
print("regra escondida:  horas = 0.400, exercícios = 0.100, viés = -7.000")

# %% [markdown]
# **O aprendizado acontecendo.** A perda cai e a acurácia sobe. A acurácia não chega a 100%: alguns alunos passaram
# "contra as chances" (ou reprovaram com tudo a favor). Nenhum modelo acerta a sorte.

# %%
fig, (eixo_perda, eixo_acc) = plt.subplots(1, 2, figsize=(12, 4))
eixo_perda.plot(historico_perda)
eixo_perda.set_xscale("symlog", linthresh=10)
eixo_perda.set_xlabel("passo")
eixo_perda.set_ylabel("entropia cruzada")
eixo_perda.set_title("A perda caindo")
eixo_perda.grid(True)

eixo_acc.plot(historico_acuracia, color="tab:green")
eixo_acc.set_xscale("symlog", linthresh=10)
eixo_acc.set_xlabel("passo")
eixo_acc.set_ylabel("acurácia")
eixo_acc.set_title("A acurácia subindo")
eixo_acc.grid(True)
plt.show()

# %% [markdown]
# (O eixo horizontal é linear até o passo 10 e logarítmico depois, para mostrar bem o começo, onde quase tudo acontece.)

# %% [markdown]
# ## 9. A fronteira de decisão
#
# Para transformar a probabilidade numa resposta, usamos um limiar (threshold): se $p \ge 0{,}5$, o modelo diz "aprovado".
# Como $\sigma(z) = 0{,}5$ exatamente quando $z = 0$, a separação entre as duas respostas é o conjunto de pontos em que
#
# $$ w_1 x_1 + w_2 x_2 + b = 0 \quad\Rightarrow\quad x_2 = -\frac{w_1 x_1 + b}{w_2} $$
#
# Isso é a equação de uma reta: a fronteira de decisão (decision boundary). De um lado, o modelo diz "aprovado"; do outro, "reprovado".
#
# **Exemplo à mão**, com a regra escondida $w = (0{,}4;\ 0{,}1)$, $b = -7$: quem estuda 10 h precisa de
# $x_2 = -(0{,}4 \cdot 10 - 7)/0{,}1 = 30$ exercícios para ficar na fronteira.
#
# Vamos ver a fronteira se formando: treinamos de novo, parando em pontos diferentes do treino. Para o efeito ficar
# visível, começamos de um chute ruim, com a fronteira inclinada para o lado errado.

# %%
horas_linha = np.linspace(0, 20, 50)

fig, eixo = plt.subplots(figsize=(8, 5.5))
desenhar_alunos(eixo, X, y)
for passos, cor in [(0, "gold"), (3, "orange"), (10, "tab:purple"), (30, "gray"), (1000, "black")]:
    w_n, b_n, _, _ = treinar_logistica(X_norm, y, taxa=0.5, passos=passos, w_inicial=[1.0, -2.0], b_inicial=1.0)
    w_p, b_p = desnormalizar_pesos(w_n, b_n, media_X, desvio_X)
    eixo.plot(horas_linha, -(w_p[0] * horas_linha + b_p) / w_p[1], color=cor, linewidth=2, label=f"fronteira após {passos} passos")
eixo.plot(horas_linha, -(0.4 * horas_linha - 7) / 0.1, color="tab:green", linestyle="--", label="fronteira da regra escondida")
eixo.set_ylim(0, 50)
eixo.set_title("A fronteira de decisão se formando")
eixo.legend(fontsize=8, loc="upper right")
eixo.grid(True)
plt.show()

# %% [markdown]
# A fronteira não é só "sim ou não": longe dela o modelo tem certeza, perto dela tem dúvida. O mapa abaixo mostra a
# probabilidade prevista em cada ponto. `np.meshgrid` monta todas as combinações de horas e exercícios de uma grade.

# %%
grade_horas, grade_exercicios = np.meshgrid(np.linspace(0, 20, 200), np.linspace(0, 50, 200))
pontos_grade = np.column_stack([grade_horas.ravel(), grade_exercicios.ravel()])
probabilidade_grade = prever_probabilidade(pontos_grade, w_final, b_final).reshape(grade_horas.shape)

fig, eixo = plt.subplots(figsize=(8, 5.5))
mapa = eixo.contourf(grade_horas, grade_exercicios, probabilidade_grade, levels=20, cmap="RdBu", alpha=0.8)
eixo.contour(grade_horas, grade_exercicios, probabilidade_grade, levels=[0.5], colors="black", linewidths=2)
desenhar_alunos(eixo, X, y)
fig.colorbar(mapa, label="probabilidade prevista de aprovação")
eixo.set_title("Probabilidade de aprovação segundo o modelo")
eixo.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 10. Avaliando: acurácia e matriz de confusão
#
# A acurácia (accuracy) é a fração de acertos. Mas ela esconde **que tipo** de erro o modelo comete.
# A matriz de confusão (confusion matrix) separa os quatro casos possíveis:
#
# | | previsto: reprovado | previsto: aprovado |
# |---|---|---|
# | **real: reprovado** | verdadeiro negativo (VN) | falso positivo (FP) |
# | **real: aprovado** | falso negativo (FN) | verdadeiro positivo (VP) |
#
# Em inglês: true negative, false positive, false negative, true positive. "Positivo" é a classe 1 (aprovado).
#
# **Exemplo à mão**, com 10 alunos:
#
# - real:     1 1 1 1 0 0 0 0 1 0
# - previsto: 1 1 1 0 0 0 1 0 1 0
#
# VP = 4 (posições 1, 2, 3 e 9), FN = 1 (posição 4), FP = 1 (posição 7), VN = 4. Acurácia = $(4 + 4)/10 = 80\%$.

# %%
def matriz_confusao(classe_prevista, y):
    """Matriz 2x2: linhas = classe real (0, 1), colunas = classe prevista (0, 1)."""
    matriz = np.zeros((2, 2), dtype=int)
    for real in [0, 1]:
        for previsto in [0, 1]:
            matriz[real, previsto] = np.sum((y == real) & (classe_prevista == previsto))
    return matriz


real_exemplo = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
previsto_exemplo = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
print(matriz_confusao(previsto_exemplo, real_exemplo))
print(f"acurácia: {acuracia(previsto_exemplo, real_exemplo):.0%}")

# %% [markdown]
# O `&` é o "e" para arrays: verdadeiro só onde as duas condições são verdadeiras. Agora, a turma inteira:

# %%
def desenhar_matriz_confusao(eixo, matriz, titulo):
    """Desenha a matriz de confusão com os números dentro de cada quadrado."""
    eixo.imshow(matriz, cmap="Blues")
    for real in [0, 1]:
        for previsto in [0, 1]:
            cor = "white" if matriz[real, previsto] > matriz.max() / 2 else "black"
            eixo.text(previsto, real, matriz[real, previsto], ha="center", va="center", fontsize=16, color=cor)
    eixo.set_xticks([0, 1], ["reprovado", "aprovado"])
    eixo.set_yticks([0, 1], ["reprovado", "aprovado"])
    eixo.set_xlabel("previsto")
    eixo.set_ylabel("real")
    eixo.set_title(titulo)


classe_prevista = (prever_probabilidade(X, w_final, b_final) >= 0.5).astype(float)
matriz = matriz_confusao(classe_prevista, y)

fig, eixo = plt.subplots(figsize=(4.5, 4))
desenhar_matriz_confusao(eixo, matriz, f"acurácia: {acuracia(classe_prevista, y):.1%}")
plt.show()

# %% [markdown]
# Um cuidado: aqui medimos os acertos nos mesmos alunos usados para treinar. É como corrigir uma prova com as
# questões que o aluno já viu. No Módulo 05 vamos separar dados de teste, nunca vistos no treino.

# %% [markdown]
# ### Comparação com o scikit-learn
#
# A biblioteca scikit-learn tem uma regressão logística pronta. Um valor de `C` grande desliga um ajuste extra
# (a regularização) que ela faz por padrão, para a comparação ser justa.

# %%
from sklearn.linear_model import LogisticRegression

modelo_pronto = LogisticRegression(C=1e6).fit(X_norm, y)
print("scikit-learn (dados normalizados):", np.round(modelo_pronto.coef_[0], 3), round(modelo_pronto.intercept_[0], 3))
print("nosso código (dados normalizados):", np.round(w_norm, 3), round(b_norm, 3))

# %% [markdown]
# Os números batem: o que a biblioteca faz por dentro é o que você acabou de escrever.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Usando a regra escondida ($w = (0{,}4;\ 0{,}1)$, $b = -7$), calcule à mão $z$ e $p$ para um aluno com
# 8 horas de estudo e 40 exercícios. Se esse aluno **reprovou**, qual é a entropia cruzada dele? Confira com `prever_probabilidade`
# e `entropia_cruzada`. (Use $e^{-0{,}2} \approx 0{,}819$ e $\log 0{,}45 \approx -0{,}799$.)
#
# **Exercício 2 (médio).** Troque o limiar de 0,5 por 0,3 e depois por 0,7. Para cada um, mostre a acurácia e a matriz de
# confusão. O que acontece com os falsos positivos e os falsos negativos? Se a escola quer **não deixar escapar** nenhum aluno
# em risco de reprovar, qual limiar é melhor?
#
# **Exercício 3 (desafio).** Treine a regressão logística usando o erro quadrático em vez da entropia cruzada.
# O gradiente passa a ser $\frac{2}{n} \sum_i (p_i - y_i)\, p_i (1 - p_i)\, x_{ij}$ (o fator $p(1-p)$ vem da derivada da sigmoide).
# Comece dos dois jeitos de um ponto ruim, $w = (-4, -4)$ e $b = 0$ nos dados normalizados, com taxa 0,5 e 300 passos.
# Compare as curvas de acurácia. Qual se recupera mais rápido?

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | classification | classificação | *Classification answers "which class?".* |
# | binary | binário | *Pass or fail is a binary outcome.* |
# | class | classe | *Each student belongs to one class.* |
# | probability | probabilidade | *The probability of passing is 0.8.* |
# | sigmoid | sigmoide | *The sigmoid squashes any number between 0 and 1.* |
# | logistic regression | regressão logística | *Logistic regression is used for classification.* |
# | cross-entropy / log loss | entropia cruzada | *Cross-entropy punishes confident mistakes.* |
# | chain rule | regra da cadeia | *The chain rule multiplies the slopes of each link.* |
# | threshold | limiar | *We predict "pass" when p is above the threshold.* |
# | decision boundary | fronteira de decisão | *The decision boundary is a straight line.* |
# | accuracy | acurácia | *The model reached 85 percent accuracy.* |
# | confusion matrix | matriz de confusão | *The confusion matrix shows the type of each error.* |
# | true positive | verdadeiro positivo | *A true positive is a correct "pass".* |
# | false positive | falso positivo | *A false positive predicts "pass" for a student who failed.* |
# | false negative | falso negativo | *A false negative misses a student who passed.* |

# %% [markdown]
# ## Resumo
#
# - Classificar é escolher uma classe; um bom classificador dá a **probabilidade** de cada classe.
# - A regressão logística calcula uma pontuação linear $z = Xw + b$ e a transforma em probabilidade com a sigmoide.
# - A entropia cruzada pune erros confiantes; com ela, o gradiente fica simples: $\frac{1}{n} X^T (p - y)$.
# - A fronteira de decisão ($z = 0$, ou seja, $p = 0{,}5$) é uma reta no espaço dos atributos.
# - Acurácia conta acertos; a matriz de confusão mostra o tipo de cada erro e muda com o limiar.
#
# **Próximo módulo:** e quando uma reta não consegue separar as classes? Vamos construir a primeira rede neural, do zero.

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
# $z = 0{,}4 \cdot 8 + 0{,}1 \cdot 40 - 7 = 3{,}2 + 4 - 7 = 0{,}2$.
# $p = \sigma(0{,}2) = 1/(1 + 0{,}819) \approx 0{,}55$.
# Se reprovou ($y = 0$): $L = -\log(1 - 0{,}55) = -\log 0{,}45 \approx 0{,}799$.

# %%
aluno_ex1 = np.array([[8.0, 40.0]])
p_ex1 = prever_probabilidade(aluno_ex1, np.array([0.4, 0.1]), -7.0)
print(f"p = {p_ex1[0]:.4f}")
print(f"entropia cruzada se reprovou: {entropia_cruzada(p_ex1, np.array([0.0])):.4f}")

# %% [markdown]
# ### Solução do Exercício 2

# %%
probabilidades = prever_probabilidade(X, w_final, b_final)

fig, eixos = plt.subplots(1, 3, figsize=(14, 4))
for eixo, limiar in zip(eixos, [0.3, 0.5, 0.7]):
    classes = (probabilidades >= limiar).astype(float)
    matriz_limiar = matriz_confusao(classes, y)
    desenhar_matriz_confusao(eixo, matriz_limiar, f"limiar {limiar}: acurácia {acuracia(classes, y):.1%}")
    print(f"limiar {limiar}: falsos positivos = {matriz_limiar[0, 1]}, falsos negativos = {matriz_limiar[1, 0]}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Com limiar baixo (0,3), o modelo diz "aprovado" com mais facilidade: aumentam os falsos positivos e diminuem os falsos negativos.
# Com limiar alto (0,7), acontece o contrário. Para não deixar escapar alunos em risco, o erro mais grave é o **falso positivo**
# (dizer "vai passar" para quem vai reprovar). Então a escola prefere o limiar **alto**, mesmo perdendo um pouco de acurácia.
# A escolha do limiar depende do custo de cada tipo de erro, não só da acurácia.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def gradientes_quadratico(X, y, w, b):
    """Gradiente do erro quadrático médio (p - y)² quando p = sigmoide(Xw + b)."""
    n = len(y)
    p = prever_probabilidade(X, w, b)
    fator = 2 * (p - y) * p * (1 - p)
    return (X.T @ fator) / n, np.sum(fator) / n


w_q, b_q = np.array([-4.0, -4.0]), 0.0
acuracia_quadratico = []
for passo in range(301):
    acuracia_quadratico.append(acuracia(prever_probabilidade(X_norm, w_q, b_q) >= 0.5, y))
    gradiente_w, gradiente_b = gradientes_quadratico(X_norm, y, w_q, b_q)
    w_q, b_q = w_q - 0.5 * gradiente_w, b_q - 0.5 * gradiente_b

_, _, _, acuracia_entropia = treinar_logistica(X_norm, y, taxa=0.5, passos=300, w_inicial=[-4.0, -4.0], b_inicial=0.0)

plt.figure(figsize=(7, 4))
plt.plot(acuracia_entropia, label="entropia cruzada")
plt.plot(acuracia_quadratico, label="erro quadrático")
plt.xlabel("passo")
plt.ylabel("acurácia")
plt.title("Recuperando-se de um começo ruim")
plt.legend()
plt.grid(True)
plt.show()
print(f"acurácia após 300 passos: entropia cruzada = {acuracia_entropia[-1]:.1%}, erro quadrático = {acuracia_quadratico[-1]:.1%}")

# %% [markdown]
# Com a entropia cruzada, o modelo sai rapidamente do começo ruim. Com o erro quadrático, ele demora muito mais:
# os alunos em que o modelo está mais errado têm $p(1-p)$ quase zero e quase não contribuem para o gradiente.
