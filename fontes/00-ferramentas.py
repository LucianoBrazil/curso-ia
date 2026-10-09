# %% [markdown]
# # Módulo 00 — Ferramentas: Python, NumPy e gráficos
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/00-ferramentas.ipynb)
#
# **Objetivo.** Aprender o mínimo de Python, NumPy e matplotlib para acompanhar o curso inteiro.
# Tudo com um exemplo de Física: um objeto em queda livre.
#
# **O que você vai construir.** Um pequeno "laboratório virtual": você vai gerar medições de uma queda,
# desenhar gráficos e, no fim, descobrir o valor da gravidade testando chutes. Esse último passo já é
# uma primeira ideia do que é "aprender" em IA.
#
# Se você já programa em Python e conhece NumPy, pode pular para o Módulo 01.

# %% [markdown]
# ## 1. A intuição: uma pedra caindo
#
# Solte uma pedra do alto de um prédio. No começo ela cai devagar. A cada segundo, fica mais rápida.
# Por isso a distância percorrida não cresce em linha reta: cresce cada vez mais depressa.
#
# Se ignorarmos a resistência do ar, a Física dá uma fórmula exata para a distância percorrida
# depois de $t$ segundos:
#
# $$ s(t) = \frac{1}{2}\, g\, t^2 $$
#
# - $s$ é a distância em metros.
# - $t$ é o tempo em segundos.
# - $g \approx 9{,}8\ \text{m/s}^2$ é a aceleração da gravidade na Terra.
#
# A velocidade nesse instante é:
#
# $$ v(t) = g\, t $$
#
# **Exemplo à mão.** Depois de $t = 2$ s:
#
# $$ s(2) = \frac{1}{2} \cdot 9{,}8 \cdot 2^2 = \frac{1}{2} \cdot 9{,}8 \cdot 4 = 19{,}6\ \text{m} $$
#
# $$ v(2) = 9{,}8 \cdot 2 = 19{,}6\ \text{m/s} $$
#
# Depois de $t = 4$ s, o tempo dobrou, mas a distância quadruplicou: $s(4) = 78{,}4$ m.
# É o efeito do $t^2$. Vamos agora fazer o computador repetir essas contas.

# %% [markdown]
# ## 2. Variáveis e tipos
#
# Uma variável (variable) é um nome que guarda um valor. O sinal `=` significa "guarde isto aqui",
# não "é igual a" como na Matemática.
#
# A função `print` mostra um valor na tela.

# %%
g = 9.8          # gravidade, em m/s² (número real: float)
t = 2            # tempo, em segundos (número inteiro: int)
planeta = "Terra"  # texto (string)

s = 0.5 * g * t**2   # ** é a potência: t**2 = t ao quadrado
print(s)

# %% [markdown]
# O resultado bate com a conta à mão: 19,6 m. (Em Python, a vírgula decimal é um ponto: `9.8`.)
#
# A função `type` mostra o tipo (type) de um valor.

# %%
print(type(g), type(t), type(planeta))

# %% [markdown]
# Para misturar texto e números, usamos uma f-string: um texto com `f` na frente,
# onde o que estiver entre chaves `{}` é calculado. O trecho `:.1f` pede uma casa decimal.

# %%
print(f"Na {planeta}, depois de {t} s, a pedra caiu {s:.1f} m.")

# %% [markdown]
# ## 3. Listas
#
# Uma lista (list) guarda vários valores em ordem, entre colchetes.
# A posição de cada item é o índice (index), e **começa em zero**.

# %%
tempos = [0, 1, 2, 3, 4]
print(tempos[0])    # primeiro item
print(tempos[2])    # terceiro item
print(tempos[-1])   # o -1 pega o último
print(len(tempos))  # len = quantidade de itens

# %% [markdown]
# Podemos acrescentar itens com `append` e pegar um pedaço da lista com `[início:fim]`.
# O fim não entra no pedaço.

# %%
tempos.append(5)
print(tempos)
print(tempos[1:4])  # índices 1, 2 e 3

# %% [markdown]
# ## 4. Funções
#
# Uma função (function) é uma receita com nome. Ela recebe entradas (os argumentos)
# e devolve uma saída com `return`. Assim não precisamos reescrever a fórmula toda vez.
#
# O `g=9.8` é um valor padrão: se você não informar `g`, a função usa 9,8.

# %%
def posicao_queda(t, g=9.8):
    """Distância percorrida (m) por um objeto em queda livre após t segundos."""
    return 0.5 * g * t**2


def velocidade_queda(t, g=9.8):
    """Velocidade (m/s) de um objeto em queda livre após t segundos."""
    return g * t


print(posicao_queda(2))
print(posicao_queda(2, g=1.62))  # na Lua, a gravidade é bem menor
print(velocidade_queda(2))

# %% [markdown]
# Atenção à indentação (indentation): os espaços no começo das linhas dizem ao Python
# o que faz parte da função. Em Python, isso não é estética; é regra.

# %% [markdown]
# ## 5. Laços: repetindo uma tarefa
#
# O laço `for` (for loop) repete um bloco de código para cada item de uma lista.
# A função `range(n)` gera os números de 0 até n-1.

# %%
for t in range(6):
    print(f"t = {t} s  ->  s = {posicao_queda(t):6.1f} m,  v = {velocidade_queda(t):5.1f} m/s")

# %% [markdown]
# Veja na tabela o que a fórmula prevê: a velocidade cresce sempre a mesma quantidade
# (9,8 m/s por segundo), mas a distância cresce cada vez mais.
#
# Também podemos construir uma lista dentro de um laço:

# %%
posicoes = []
for t in tempos:
    posicoes.append(posicao_queda(t))
print(posicoes)

# %% [markdown]
# ## 6. Condicionais: tomando decisões
#
# O `if` (condicional, conditional) executa um bloco só se uma condição for verdadeira.
# O `elif` testa outra condição, e o `else` cobre todo o resto.
#
# Exemplo: um prédio de 45 m. Em cada instante, a pedra já chegou ao chão?

# %%
altura_predio = 45.0

for t in range(5):
    s = posicao_queda(t)
    if s >= altura_predio:
        print(f"t = {t} s: já chegou ao chão.")
    elif s >= altura_predio / 2:
        print(f"t = {t} s: passou da metade ({s:.1f} m).")
    else:
        print(f"t = {t} s: ainda na metade de cima ({s:.1f} m).")

# %% [markdown]
# ## 7. NumPy: contas com muitos números de uma vez
#
# Até aqui calculamos um tempo por vez, com laços. Em IA trabalhamos com milhares ou
# milhões de números. Para isso existe o NumPy, uma biblioteca (library) de Matemática.
#
# A peça central do NumPy é o array: parecido com uma lista, mas feito para contas.
# A grande vantagem é a vetorização (vectorization): uma operação aplicada ao array
# inteiro de uma só vez, sem escrever o laço.

# %%
import numpy as np   # "np" é o apelido que todo mundo usa

t = np.array([0, 1, 2, 3, 4, 5])
print(t * 2)     # cada elemento multiplicado por 2
print(t ** 2)    # cada elemento ao quadrado

# %% [markdown]
# Compare: com uma lista comum, `[0, 1, 2] * 2` repete a lista (`[0, 1, 2, 0, 1, 2]`).
# Com um array, multiplica cada número. É o comportamento que queremos na Matemática.
#
# A melhor parte: a nossa função `posicao_queda` funciona com arrays sem nenhuma mudança.

# %%
print(posicao_queda(t))

# %% [markdown]
# ### Criando arrays
#
# - `np.linspace(a, b, n)`: n números igualmente espaçados de a até b (incluindo b).
# - `np.arange(a, b, passo)`: de a até b (sem incluir b), de passo em passo.
# - `np.zeros(n)`: n zeros.

# %%
t_fino = np.linspace(0, 3, 7)
print(t_fino)
print(np.arange(0, 1, 0.25))
print(np.zeros(4))

# %% [markdown]
# ### Forma, soma, média e máximo
#
# O atributo `shape` (forma) diz o tamanho do array. Funções como `sum`, `mean` (média)
# e `max` resumem o array inteiro.

# %%
s = posicao_queda(t_fino)
print("forma:", s.shape)
print("soma:", s.sum())
print("média:", s.mean())
print("máximo:", s.max())

# %% [markdown]
# ### Máscaras booleanas: filtrando sem `if`
#
# Uma comparação com um array devolve um array de `True`/`False`. Esse array, chamado
# máscara booleana (boolean mask), serve para escolher só os elementos que interessam.

# %%
mascara = s > 10
print(mascara)
print(t_fino[mascara])  # os instantes em que a pedra já caiu mais de 10 m

# %% [markdown]
# ### Por que vetorizar? Uma corrida
#
# Vamos calcular a posição em um milhão de instantes, primeiro com um laço,
# depois com NumPy, e medir o tempo de cada um.

# %%
import time

muitos_tempos = np.linspace(0, 10, 1_000_000)
lista_tempos = muitos_tempos.tolist()   # a mesma coisa, como lista comum

inicio = time.perf_counter()
resultado_laco = [0.5 * 9.8 * x * x for x in lista_tempos]
tempo_laco = time.perf_counter() - inicio

inicio = time.perf_counter()
resultado_numpy = 0.5 * 9.8 * muitos_tempos * muitos_tempos
tempo_numpy = time.perf_counter() - inicio

print(f"laço:  {tempo_laco:.3f} s")
print(f"NumPy: {tempo_numpy:.4f} s")
print(f"NumPy foi cerca de {tempo_laco / tempo_numpy:.0f} vezes mais rápido.")

# %% [markdown]
# (A linha `[... for x in ...]` é uma forma curta de montar uma lista com um laço.)
#
# Os números exatos variam de máquina para máquina; o NumPy costuma ser de algumas a dezenas de vezes mais rápido.
# Por isso, no curso todo, vamos preferir operações vetorizadas.

# %% [markdown]
# ## 8. Gráficos com matplotlib
#
# O matplotlib desenha gráficos. O comando básico é `plt.plot(x, y)`.
# Sempre dê nome aos eixos, com a unidade.

# %%
import matplotlib.pyplot as plt

t_grafico = np.linspace(0, 3, 100)

plt.figure(figsize=(7, 4))
plt.plot(t_grafico, posicao_queda(t_grafico), label="Terra (g = 9,8)")
plt.plot(t_grafico, posicao_queda(t_grafico, g=1.62), label="Lua (g = 1,62)")
plt.xlabel("tempo (s)")
plt.ylabel("distância percorrida (m)")
plt.title("Queda livre")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# A curva da Terra é uma parábola: é o $t^2$ da fórmula aparecendo no desenho.

# %% [markdown]
# ## 9. Medições de verdade têm ruído
#
# Num laboratório real, nenhuma medição é perfeita. O cronômetro atrasa, a régua tem
# limites. Chamamos esse erro aleatório de ruído (noise).
#
# Vamos simular um experimento. O NumPy sorteia números aleatórios com um gerador.
# Fixamos a semente (seed) em 42: assim os "sorteios" saem iguais toda vez que você rodar.
#
# `rng.normal(0, 1.5, n)` sorteia n números em torno de 0, quase todos entre -4,5 e +4,5.

# %%
rng = np.random.default_rng(42)

t_medido = np.linspace(0.2, 3.0, 15)
ruido = rng.normal(0, 1.5, size=t_medido.shape)
s_medido = posicao_queda(t_medido) + ruido

plt.figure(figsize=(7, 4))
plt.scatter(t_medido, s_medido, color="tab:red", label="medições (com ruído)")
plt.plot(t_grafico, posicao_queda(t_grafico), label="fórmula (g = 9,8)")
plt.xlabel("tempo (s)")
plt.ylabel("distância (m)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# `plt.scatter` desenha pontos soltos; `plt.plot` liga os pontos com uma linha.
# As medições ficam perto da curva, mas não exatamente em cima dela.

# %% [markdown]
# ## 10. Uma prévia do que é "aprender"
#
# Agora inverta o problema. Imagine que você **não sabe** o valor de $g$.
# Só tem as medições. Como descobrir $g$?
#
# Uma ideia simples: chutar vários valores de $g$ e, para cada chute, medir o quanto
# a fórmula erra. Vamos medir o erro assim: para cada ponto, calcule a diferença entre
# a previsão e a medição, eleve ao quadrado (para que erros negativos não cancelem os
# positivos) e tire a média. Isso se chama erro quadrático médio (mean squared error, MSE):
#
# $$ \text{erro}(g) = \frac{1}{n} \sum_{i=1}^{n} \big( s_{\text{previsto},i} - s_{\text{medido},i} \big)^2 $$
#
# O símbolo $\sum$ significa "some todos os termos", de $i = 1$ até $n$.
#
# **Exemplo à mão**, com dois pontos só: medições de 5 m e 19 m em $t = 1$ s e $t = 2$ s.
# Com o chute $g = 10$, a fórmula prevê 5 m e 20 m. As diferenças são 0 e 1.
#
# $$ \text{erro}(10) = \frac{0^2 + 1^2}{2} = 0{,}5 $$
#
# O melhor chute é o que deixa o erro menor. No Módulo 01 você vai entender esse erro a fundo.

# %%
def erro_quadratico_medio(previsto, medido):
    """Média dos quadrados das diferenças entre previsão e medição."""
    return np.mean((previsto - medido) ** 2)


# Conferindo o exemplo feito à mão
print(erro_quadratico_medio(np.array([5.0, 20.0]), np.array([5.0, 19.0])))

# %% [markdown]
# Agora testamos 200 chutes de $g$, entre 5 e 15, e guardamos o erro de cada um.

# %%
chutes_g = np.linspace(5, 15, 200)
erros = []
for g_chute in chutes_g:
    previsto = posicao_queda(t_medido, g=g_chute)
    erros.append(erro_quadratico_medio(previsto, s_medido))
erros = np.array(erros)

melhor_g = chutes_g[np.argmin(erros)]   # argmin = posição do menor valor
print(f"Melhor chute: g = {melhor_g:.2f} m/s²")

# %% [markdown]
# O gráfico abaixo mostra o erro para cada chute. O ponto mais baixo da curva é o
# melhor valor de $g$. Encontrar esse ponto mais baixo, de forma esperta, é o coração
# do aprendizado de máquina (machine learning).

# %%
plt.figure(figsize=(7, 4))
plt.plot(chutes_g, erros)
plt.axvline(melhor_g, color="tab:red", linestyle="--", label=f"melhor g = {melhor_g:.2f}")
plt.axvline(9.8, color="gray", linestyle=":", label="valor real 9,8")
plt.xlabel("chute para g (m/s²)")
plt.ylabel("erro quadrático médio (m²)")
plt.title("Quanto a fórmula erra para cada chute de g")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# O melhor chute fica perto de 9,8, mas não exatamente, por causa do ruído.
# Com mais medições, ele chegaria mais perto. Repare: o computador "descobriu" a
# gravidade a partir dos dados, sem ninguém dizer o valor a ele.

# %% [markdown]
# ## Mão na massa
#
# Escreva o código em células novas logo abaixo (menu "+ Código" no Colab).
#
# **Exercício 1 (fácil).** Escreva uma função `tempo_de_queda(altura, g=9.8)` que devolve
# quanto tempo um objeto leva para cair de uma certa altura. Dica: isole $t$ em
# $h = \frac{1}{2} g t^2$, o que dá $t = \sqrt{2h/g}$. Use `np.sqrt`.
# Teste com o prédio de 45 m (a resposta é cerca de 3 s).
#
# **Exercício 2 (médio).** Usando NumPy, sem laço, calcule a posição de 0 a 5 s, de 0,5 em 0,5 s.
# Depois use uma máscara booleana para mostrar só os instantes em que a pedra ainda não
# chegou ao chão de um prédio de 80 m.
#
# **Exercício 3 (desafio).** Faça um gráfico com a queda livre em três lugares:
# Terra (9,8), Lua (1,62) e Júpiter (24,8), de 0 a 3 s, com legenda. Em seguida, repita o
# experimento da seção 10 com 100 medições em vez de 15 (use outra semente, por exemplo 7).
# O melhor chute de $g$ chegou mais perto de 9,8?

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | variable | variável | *The variable `g` stores the gravity.* |
# | list | lista | *A list keeps items in order.* |
# | index | índice | *Python indexes start at zero.* |
# | function | função | *This function returns the position.* |
# | loop | laço | *The loop runs six times.* |
# | conditional | condicional | *A conditional checks if the stone hit the ground.* |
# | array | array (vetor) | *NumPy arrays are fast.* |
# | vectorization | vetorização | *Vectorization removes slow loops.* |
# | boolean mask | máscara booleana | *Use a boolean mask to filter the data.* |
# | seed | semente | *Fix the seed to get the same results.* |
# | noise | ruído | *Real measurements always have noise.* |
# | plot | gráfico / desenhar | *Plot the distance against time.* |
# | mean squared error | erro quadrático médio | *A smaller mean squared error means a better fit.* |

# %% [markdown]
# ## Resumo
#
# - Variáveis guardam valores; listas guardam vários valores em ordem, com índice a partir de zero.
# - Funções empacotam uma fórmula; `for` repete e `if` decide.
# - Arrays do NumPy fazem contas no conjunto inteiro de uma vez (vetorização), muito mais rápido que laços.
# - O matplotlib desenha curvas (`plot`) e pontos (`scatter`); sempre nomeie os eixos.
# - Testando chutes e medindo o erro, descobrimos $g$ a partir de dados com ruído: uma prévia de "aprender".
#
# **Próximo módulo:** em vez de testar chutes um por um, vamos ajustar uma reta seguindo a inclinação do erro, com o gradiente descendente.

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
def tempo_de_queda(altura, g=9.8):
    """Tempo (s) para um objeto cair de uma altura (m), sem resistência do ar."""
    return np.sqrt(2 * altura / g)


print(f"{tempo_de_queda(45):.2f} s")

# %% [markdown]
# ### Solução do Exercício 2

# %%
t_ex2 = np.arange(0, 5.5, 0.5)       # 5.5 para que o 5 entre
s_ex2 = posicao_queda(t_ex2)
ainda_caindo = s_ex2 < 80
print("instantes:", t_ex2[ainda_caindo])
print("posições: ", s_ex2[ainda_caindo])

# %% [markdown]
# ### Solução do Exercício 3

# %%
t_ex3 = np.linspace(0, 3, 100)
plt.figure(figsize=(7, 4))
for nome, g_local in [("Terra", 9.8), ("Lua", 1.62), ("Júpiter", 24.8)]:
    plt.plot(t_ex3, posicao_queda(t_ex3, g=g_local), label=f"{nome} (g = {g_local})")
plt.xlabel("tempo (s)")
plt.ylabel("distância percorrida (m)")
plt.legend()
plt.grid(True)
plt.show()

rng7 = np.random.default_rng(7)
t_100 = np.linspace(0.2, 3.0, 100)
s_100 = posicao_queda(t_100) + rng7.normal(0, 1.5, size=t_100.shape)
erros_100 = np.array([erro_quadratico_medio(posicao_queda(t_100, g=gc), s_100) for gc in chutes_g])
print(f"Melhor chute com 100 medições: g = {chutes_g[np.argmin(erros_100)]:.2f} m/s²")
