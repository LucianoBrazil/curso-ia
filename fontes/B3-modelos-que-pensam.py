# %% [markdown]
# # Módulo B3 (bônus) — Na fronteira: modelos que pensam antes de responder
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B3-modelos-que-pensam.ipynb)
#
# **Objetivo.** Ver, em miniatura e com experimentos de verdade, as ideias por trás dos modelos de IA mais recentes: pensar em voz alta,
# gastar mais computação na hora de responder, aprender com recompensas e usar ferramentas.
#
# **O que você vai construir.** Partindo do Transformer de caracteres do Módulo 10, você vai treinar modelos que somam números de três
# algarismos: um que responde direto e outro que escreve a conta armada antes. Depois, vai fazer o modelo votar entre várias tentativas,
# melhorá-lo só com "acertou/errou" e montar um pequeno agente que usa uma calculadora.
#
# Não é um projeto completo, e sim uma amostra prática da direção atual da IA.

# %% [markdown]
# ## 1. A intuição: pensar antes de responder
#
# Quanto é 347 + 589? Quase ninguém responde de cabeça, de uma vez. Armamos a conta: 7 + 9 = 16, escreve 6 e vai um; 4 + 8 + 1 = 13,
# escreve 3 e vai um; 3 + 5 + 1 = 9. Resposta: 936. O papel funciona como rascunho: guardamos resultados intermediários e cada passo fica fácil.
#
# Lembre como o Transformer do Módulo 10 escreve: cada token sai de **uma** passada pela rede, com um número fixo de camadas. Se o modelo
# precisa dar a resposta logo de cara, toda a conta tem de caber nessas poucas camadas. Se ele puder escrever os passos antes, cada token
# escrito é mais uma passada pela rede, e o próprio texto vira o rascunho.
#
# Essa é a ideia dos modelos de raciocínio (reasoning models), como a série o1 da OpenAI, o DeepSeek-R1 e o Claude com pensamento
# estendido: antes de responder, eles escrevem uma cadeia de pensamento (chain of thought), às vezes longa.
#
# Para estudar isso, precisamos de uma tarefa cuja resposta dê para **conferir automaticamente**. Somar números de três algarismos é perfeita:
# o Python sabe a resposta certa, e podemos medir o acerto sem ninguém corrigir à mão.
#
# **Sobre o tempo.** Alguns treinos deste módulo levariam vários minutos no Colab. Por isso, o notebook carrega modelos já treinados da pasta
# `dados/` do repositório. Se quiser treinar tudo do zero, mude a variável abaixo para `True` (leva uns 10 minutos no Colab gratuito);
# os resultados vão variar um pouco, mas as conclusões são as mesmas.

# %%
import os
import time
import urllib.request
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

TREINAR_DO_ZERO = False   # True: treina todos os modelos aqui; False: carrega os pesos já treinados

# %% [markdown]
# ## 2. A tarefa: somar números de três algarismos
#
# **A conta armada, em símbolos.** Chamamos de $a_i$ e $b_i$ os algarismos das duas parcelas na posição $i$ ($i = 0$ para as unidades,
# $1$ para as dezenas, $2$ para as centenas) e de $c_i$ o "vai um" que chega a essa posição ($c_0 = 0$). Em cada posição:
#
# $$ s_i = a_i + b_i + c_i \qquad d_i = s_i \bmod 10 \qquad c_{i+1} = \left\lfloor s_i / 10 \right\rfloor $$
#
# $s_i \bmod 10$ é o resto da divisão por 10 (o algarismo que escrevemos) e $\lfloor s_i / 10 \rfloor$ é a parte inteira da divisão
# (o "vai um", que só pode ser 0 ou 1).
#
# **Exemplo à mão**, 347 + 589:
#
# | posição | $a_i + b_i + c_i$ | $s_i$ | escreve $d_i$ | vai $c_{i+1}$ |
# |---|---|---|---|---|
# | unidades | 7 + 9 + 0 | 16 | 6 | 1 |
# | dezenas | 4 + 8 + 1 | 13 | 3 | 1 |
# | centenas | 3 + 5 + 1 | 9 | 9 | 0 |
#
# Resposta: o último "vai um" seguido de $d_2 d_1 d_0$, ou seja, 0936. Para simplificar, as respostas sempre terão 4 algarismos
# (completando com zero à esquerda) e terminarão com um ponto.
#
# **Os dois formatos de texto** que os modelos vão aprender:
#
# - **direto:** `347+589=0936.`
# - **com pensamento:** `347+589=7+9+0=16,4+8+1=13,3+5+1=09,0936.` (cada passo é "algarismo + algarismo + vai um = soma de dois algarismos")

# %%
VOCABULARIO = "0123456789+=,.()[]CAL"
TAMANHO_VOCABULARIO = len(VOCABULARIO)
TAMANHO_CONTEXTO = 48
indice_de = {caractere: i for i, caractere in enumerate(VOCABULARIO)}


def codificar(texto):
    """Texto -> lista de índices (Módulo 08)."""
    return [indice_de[c] for c in texto]


def decodificar(indices):
    """Índices -> texto (Módulo 08)."""
    return "".join(VOCABULARIO[int(i)] for i in indices)


def pergunta(a, b):
    """A pergunta, sempre com 3 algarismos em cada parcela."""
    return f"{a:03d}+{b:03d}="


def resposta_direta(a, b):
    """Só a resposta, com 4 algarismos e um ponto final."""
    return f"{a + b:04d}."


def resposta_com_pensamento(a, b):
    """Os passos da conta armada, das unidades para as centenas, e depois a resposta."""
    algarismos_a, algarismos_b = f"{a:03d}", f"{b:03d}"
    passos, vai_um = [], 0
    for posicao in [2, 1, 0]:          # unidades, dezenas, centenas
        soma = int(algarismos_a[posicao]) + int(algarismos_b[posicao]) + vai_um
        passos.append(f"{algarismos_a[posicao]}+{algarismos_b[posicao]}+{vai_um}={soma:02d}")
        vai_um = soma // 10
    return ",".join(passos) + "," + resposta_direta(a, b)


print(pergunta(347, 589) + resposta_direta(347, 589))
print(pergunta(347, 589) + resposta_com_pensamento(347, 589))

# %% [markdown]
# **Os dados.** Sorteamos três grupos de contas, sem repetição entre eles (Módulo 05): 1000 para treino, 300 para validação e 500 para teste.
# Repare: só **1000** contas de exemplo, entre um milhão de contas possíveis ($1000 \times 1000$).

# %%
def sortear_pares(quantidade, gerador, excluir=()):
    """Sorteia pares (a, b) distintos, de 0 a 999, fora do conjunto excluir."""
    pares, vistos = [], set(excluir)
    while len(pares) < quantidade:
        par = (int(gerador.integers(0, 1000)), int(gerador.integers(0, 1000)))
        if par not in vistos:
            vistos.add(par)
            pares.append(par)
    return pares


gerador_dados = np.random.default_rng(42)
pares_teste = sortear_pares(500, gerador_dados)
pares_validacao = sortear_pares(300, gerador_dados, excluir=pares_teste)
pares_treino = sortear_pares(1000, gerador_dados, excluir=pares_teste + pares_validacao)
fora_do_treino = set(pares_teste) | set(pares_validacao)
print("exemplos de treino:", pares_treino[:5])

# %% [markdown]
# Para treinar, cada conta vira uma sequência de índices. Como no Módulo 10, o alvo é a mesma sequência deslocada uma posição. Uma diferença:
# não faz sentido cobrar do modelo que adivinhe a **pergunta**, que é sorteada. Por isso, nas posições da pergunta, o alvo recebe o valor
# especial −100, que a `nn.CrossEntropyLoss` ignora (`ignore_index=-100`). O modelo só é avaliado no que vem depois do "=".

# %%
def montar_exemplos(pares, escrever_resposta):
    """Entradas e alvos para o treino; o alvo vale -100 (ignorado) nas posições da pergunta."""
    textos = [pergunta(a, b) + escrever_resposta(a, b) for a, b in pares]
    indices = torch.tensor([codificar(texto) for texto in textos])
    entradas, alvos = indices[:, :-1], indices[:, 1:].clone()
    alvos[:, :len(pergunta(0, 0)) - 1] = -100
    return entradas, alvos


entradas_exemplo, alvos_exemplo = montar_exemplos(pares_treino[:1], resposta_direta)
print("entrada:", entradas_exemplo[0].tolist())
print("alvo:   ", alvos_exemplo[0].tolist())

# %% [markdown]
# ## 3. O modelo: o Transformer do Módulo 10
#
# As classes abaixo são as do Módulo 10, com um ajuste: o tamanho do vocabulário e do contexto viram parâmetros. O modelo tem 3 blocos,
# 4 cabeças de atenção e vetores de tamanho 64.

# %%
class CabecaAtencao(nn.Module):
    """Vimos isso no Módulo 10: uma cabeça de autoatenção causal."""

    def __init__(self, dim_modelo, dim_cabeca):
        super().__init__()
        self.consulta = nn.Linear(dim_modelo, dim_cabeca, bias=False)
        self.chave = nn.Linear(dim_modelo, dim_cabeca, bias=False)
        self.valor = nn.Linear(dim_modelo, dim_cabeca, bias=False)

    def forward(self, x):
        n_posicoes = x.shape[1]
        Q, K, V = self.consulta(x), self.chave(x), self.valor(x)
        pontuacoes = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5
        mascara = torch.tril(torch.ones(n_posicoes, n_posicoes))
        pontuacoes = pontuacoes.masked_fill(mascara == 0, float("-inf"))
        return torch.softmax(pontuacoes, dim=-1) @ V


class AtencaoVariasCabecas(nn.Module):
    """Vimos isso no Módulo 10."""

    def __init__(self, dim_modelo, n_cabecas):
        super().__init__()
        self.cabecas = nn.ModuleList([CabecaAtencao(dim_modelo, dim_modelo // n_cabecas) for _ in range(n_cabecas)])
        self.mistura = nn.Linear(dim_modelo, dim_modelo)

    def forward(self, x):
        return self.mistura(torch.cat([cabeca(x) for cabeca in self.cabecas], dim=-1))


class BlocoTransformer(nn.Module):
    """Vimos isso no Módulo 10: atenção e rede densa, com normalização e conexões residuais."""

    def __init__(self, dim_modelo, n_cabecas):
        super().__init__()
        self.normalizacao_1 = nn.LayerNorm(dim_modelo)
        self.atencao = AtencaoVariasCabecas(dim_modelo, n_cabecas)
        self.normalizacao_2 = nn.LayerNorm(dim_modelo)
        self.rede_densa = nn.Sequential(
            nn.Linear(dim_modelo, 4 * dim_modelo), nn.ReLU(), nn.Linear(4 * dim_modelo, dim_modelo))

    def forward(self, x):
        x = x + self.atencao(self.normalizacao_1(x))
        return x + self.rede_densa(self.normalizacao_2(x))


class MiniTransformer(nn.Module):
    """Vimos isso no Módulo 10; agora o vocabulário e o contexto são parâmetros."""

    def __init__(self, tamanho_vocabulario, tamanho_contexto, dim_modelo=64, n_cabecas=4, n_blocos=3):
        super().__init__()
        self.embedding_token = nn.Embedding(tamanho_vocabulario, dim_modelo)
        self.embedding_posicao = nn.Embedding(tamanho_contexto, dim_modelo)
        self.blocos = nn.Sequential(*[BlocoTransformer(dim_modelo, n_cabecas) for _ in range(n_blocos)])
        self.normalizacao_final = nn.LayerNorm(dim_modelo)
        self.saida = nn.Linear(dim_modelo, tamanho_vocabulario)

    def forward(self, indices):
        x = self.embedding_token(indices) + self.embedding_posicao(torch.arange(indices.shape[1]))
        return self.saida(self.normalizacao_final(self.blocos(x)))


def criar_modelo():
    """Um MiniTransformer novo, com o vocabulário e o contexto deste módulo."""
    return MiniTransformer(TAMANHO_VOCABULARIO, TAMANHO_CONTEXTO)


print(f"{sum(p.numel() for p in criar_modelo().parameters())} parâmetros")

# %% [markdown]
# **Gerar e conferir.** A função `gerar` escreve token por token (Módulo 10). Com temperatura 0 ela escolhe sempre o token mais provável:
# é a decodificação gulosa (greedy decoding). Com temperatura maior que 0, ela sorteia (Módulo 08). Como os formatos têm tamanho fixo,
# geramos as respostas de muitas perguntas ao mesmo tempo. Depois, `extrair_resposta` pega os 4 algarismos finais e o Python confere.

# %%
@torch.no_grad()
def gerar(modelo, sequencias, n_tokens, temperatura=0.0, gerador=None):
    """Acrescenta n_tokens a cada sequência; guloso se temperatura == 0, sorteio caso contrário."""
    for _ in range(n_tokens):
        logits = modelo(sequencias)[:, -1]
        if temperatura == 0:
            proximo = logits.argmax(dim=-1)
        else:
            probabilidades = torch.softmax(logits / temperatura, dim=-1)
            proximo = torch.multinomial(probabilidades, 1, generator=gerador)[:, 0]
        sequencias = torch.cat([sequencias, proximo[:, None]], dim=1)
    return sequencias


def responder(modelo, pares, escrever_resposta, temperatura=0.0, gerador=None):
    """O texto que o modelo escreve depois de cada pergunta (com o mesmo tamanho do formato)."""
    prefixos = torch.tensor([codificar(pergunta(a, b)) for a, b in pares])
    saidas = gerar(modelo, prefixos, len(escrever_resposta(0, 0)), temperatura, gerador)
    return [decodificar(saida[prefixos.shape[1]:]) for saida in saidas]


def extrair_resposta(texto):
    """Os 4 algarismos antes do ponto final (ou '????' se o formato estiver errado)."""
    return texto[-5:-1] if texto.endswith(".") else "????"


def taxa_de_acerto(textos, pares):
    """Fração de respostas finais certas."""
    return float(np.mean([extrair_resposta(texto) == f"{a + b:04d}" for texto, (a, b) in zip(textos, pares)]))

# %% [markdown]
# ## 4. Pensar em voz alta
#
# Vamos treinar dois modelos iguais, com as **mesmas 1000 contas**, pelo mesmo número de passos (1500, lotes de 64): um no formato direto,
# outro no formato com pensamento. A cada tanto, medimos o acerto em contas do treino e em contas novas (validação).
#
# Durante o treino do modelo com pensamento, também guardamos uma cópia dele num ponto em que ainda erra bastante (acerto de validação
# perto de 70%). Ela vai servir nas seções 5 e 6, como um "aluno no meio do aprendizado".

# %%
def treinar(escrever_resposta, passos=1500, tamanho_lote=64, taxa=1e-3, semente=0, guardar_meio=False):
    """Treina um MiniTransformer nas contas de treino e devolve o modelo, o histórico e (opcional) uma cópia do meio do treino."""
    torch.manual_seed(semente)
    modelo = criar_modelo()
    entradas, alvos = montar_exemplos(pares_treino, escrever_resposta)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=0.01)
    funcao_perda = nn.CrossEntropyLoss(ignore_index=-100)
    gerador = torch.Generator().manual_seed(semente)
    historico = {"passo": [], "acerto_treino": [], "acerto_validacao": []}
    candidatos_meio = []
    for passo in range(passos + 1):
        if passo % 50 == 0 and (passo <= 500 or passo % 250 == 0):
            acerto_treino = taxa_de_acerto(responder(modelo, pares_treino[:200], escrever_resposta), pares_treino[:200])
            acerto_validacao = taxa_de_acerto(responder(modelo, pares_validacao[:200], escrever_resposta), pares_validacao[:200])
            historico["passo"].append(passo)
            historico["acerto_treino"].append(acerto_treino)
            historico["acerto_validacao"].append(acerto_validacao)
            if guardar_meio and passo <= 500:
                copia = {nome: valor.clone() for nome, valor in modelo.state_dict().items()}
                candidatos_meio.append((abs(acerto_validacao - 0.7), passo, copia))
        if passo == passos:
            break
        lote = torch.randint(0, len(entradas), (tamanho_lote,), generator=gerador)
        perda = funcao_perda(modelo(entradas[lote]).reshape(-1, TAMANHO_VOCABULARIO), alvos[lote].reshape(-1))
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    meio = min(candidatos_meio, key=lambda candidato: candidato[0])[1:] if guardar_meio else None
    return modelo, historico, meio


NOME_PESOS = "b3_pesos.pt"
CAMINHO_PESOS = "../dados/" + NOME_PESOS
URL_PESOS = "https://raw.githubusercontent.com/LucianoBrazil/curso-ia/main/dados/" + NOME_PESOS


def obter_pesos():
    """Lê os pesos de dados/ ou, no Colab, baixa do GitHub. weights_only=True: carrega só números, nunca código."""
    caminho = CAMINHO_PESOS if os.path.exists(CAMINHO_PESOS) else NOME_PESOS
    if not os.path.exists(caminho):
        urllib.request.urlretrieve(URL_PESOS, caminho)
    return torch.load(caminho, weights_only=True)


def modelo_com_pesos(estado):
    """Cria um modelo e coloca nele os pesos dados."""
    modelo = criar_modelo()
    modelo.load_state_dict(estado)
    return modelo


inicio = time.perf_counter()
if TREINAR_DO_ZERO:
    modelo_direto, historico_direto, _ = treinar(resposta_direta)
    modelo_pensamento, historico_pensamento, (passo_meio, pesos_meio) = treinar(resposta_com_pensamento, guardar_meio=True)
else:
    pesos = obter_pesos()
    modelo_direto, historico_direto = modelo_com_pesos(pesos["direto"]), pesos["historico_direto"]
    modelo_pensamento, historico_pensamento = modelo_com_pesos(pesos["pensamento"]), pesos["historico_pensamento"]
    passo_meio, pesos_meio = pesos["passo_meio"], pesos["pensamento_meio"]
modelo_meio = modelo_com_pesos(pesos_meio)
print(f"pronto em {time.perf_counter() - inicio:.0f} s; cópia do meio do treino: passo {passo_meio}")

# %% [markdown]
# **O aprendizado acontecendo**: acerto nas contas de treino (à esquerda) e em contas novas (à direita).

# %%
fig, (eixo_treino, eixo_validacao) = plt.subplots(1, 2, figsize=(13, 4.5))
for historico, nome in [(historico_direto, "direto"), (historico_pensamento, "com pensamento")]:
    eixo_treino.plot(historico["passo"], historico["acerto_treino"], marker="o", label=nome)
    eixo_validacao.plot(historico["passo"], historico["acerto_validacao"], marker="o", label=nome)
for eixo, titulo in [(eixo_treino, "Contas do treino (já vistas)"), (eixo_validacao, "Contas novas (validação)")]:
    eixo.set_xlabel("passo de treino")
    eixo.set_ylabel("taxa de acerto")
    eixo.set_ylim(-0.03, 1.03)
    eixo.set_title(titulo)
    eixo.legend()
    eixo.grid(True)
plt.show()

# %% [markdown]
# **A prova final**, nas 500 contas de teste, que nenhum modelo viu.

# %%
textos_direto = responder(modelo_direto, pares_teste, resposta_direta)
textos_pensamento = responder(modelo_pensamento, pares_teste, resposta_com_pensamento)
print(f"acerto no teste: direto {taxa_de_acerto(textos_direto, pares_teste):.1%}   "
      f"com pensamento {taxa_de_acerto(textos_pensamento, pares_teste):.1%}")
print("\nalgumas contas em que o modelo direto errou:")
erros_direto = [i for i, (a, b) in enumerate(pares_teste) if extrair_resposta(textos_direto[i]) != f"{a + b:04d}"]
for i in erros_direto[:4]:
    a, b = pares_teste[i]
    certo = f"{a + b:04d}"
    print(f"{pergunta(a, b)}  certo: {certo}   direto: {textos_direto[i]}")
    print(f"{' ' * 8}  com pensamento: {textos_pensamento[i]}  ({'certo' if extrair_resposta(textos_pensamento[i]) == certo else 'errado'})")

# %% [markdown]
# O modelo direto **decorou** o treino: acerta todas as contas que viu, mas erra quase metade das contas novas. É o sobreajuste do Módulo 05:
# ele aprendeu a regra só em parte e completou o resto decorando. O modelo com pensamento acerta quase tudo, inclusive contas que nunca viu:
# ele aprendeu o **algoritmo**. Repare nos erros do direto: muitas vezes é o "vai um" que sai errado, justamente a parte que exige
# encadear vários passos.
#
# **Por que tanta diferença?**
#
# - **Problemas pequenos em vez de um grande.** Cada passo da conta armada depende só de dois algarismos e de um "vai um": são
#   $10 \times 10 \times 2 = 200$ casos possíveis, e cada um aparece em média umas 15 vezes nas 1000 contas (são 3000 passos). Já o primeiro algarismo
#   da resposta direta depende dos seis algarismos da pergunta: um milhão de combinações, das quais o treino mostra só 1000. Com tão poucos
#   exemplos, decorar é mais fácil do que descobrir a regra.
# - **Mais computação por resposta.** O modelo direto tem 3 blocos de processamento por algarismo da resposta. O que pensa usa 32 tokens,
#   cada um com uma passada completa pela rede, e guarda os resultados parciais no texto, como num rascunho.
#
# Em 2021, pesquisadores mostraram esse efeito com adições bem mais longas (Nye e colegas, *Show Your Work: Scratchpads for Intermediate
# Computation*). Três cuidados para não exagerar. O resultado do modelo direto varia bastante com o sorteio das contas e dos pesos
# iniciais: nos testes feitos ao preparar este módulo, ficou entre menos de 10% e pouco mais de 50% nas contas novas, enquanto o modelo com
# pensamento ficou sempre perto de 100%. Com **muito** mais exemplos e treino, o modelo direto também aprende esta tarefa simples (num teste
# com 3000 contas, ele passou de 95%). E pensar custa: a resposta com pensamento tem 32 tokens, contra 5 da direta.

# %% [markdown]
# ## 5. Pensar mais na hora de responder: autoconsistência
#
# Até aqui, cada pergunta recebeu **uma** resposta. E se o modelo tentasse várias vezes, sorteando (temperatura maior que zero), e ficássemos
# com a resposta mais frequente? É como pedir a mesma conta a vários alunos e confiar na resposta da maioria. Essa técnica se chama
# autoconsistência (self-consistency), proposta em 2022 (Wang e colegas). A intuição: quando o modelo acerta, chega sempre ao mesmo
# resultado; quando erra, erra de jeitos diferentes, e os votos errados se espalham.
#
# **Exemplo à mão.** Cinco tentativas para 347 + 589: 0936, 0926, 0936, 0836, 0936. A mais votada é 0936, com 3 votos, e está certa,
# embora duas tentativas tenham errado.
#
# Usamos a cópia do meio do treino (`modelo_meio`), que ainda erra bastante, em 200 perguntas do teste. Para cada pergunta, sorteamos
# 21 tentativas com temperatura 0,7 e medimos o acerto da votação usando as primeiras 1, 3, 5, ..., 21 tentativas.

# %%
def votar(respostas):
    """A resposta mais frequente (voto da maioria)."""
    return Counter(respostas).most_common(1)[0][0]


print("votação do exemplo à mão:", votar(["0936", "0926", "0936", "0836", "0936"]))

perguntas_votacao = pares_teste[:200]
N_TENTATIVAS = 21
repetidas = [par for par in perguntas_votacao for _ in range(N_TENTATIVAS)]
inicio = time.perf_counter()
textos_sorteados = responder(modelo_meio, repetidas, resposta_com_pensamento, temperatura=0.7,
                             gerador=torch.Generator().manual_seed(0))
tentativas = np.array([extrair_resposta(texto) for texto in textos_sorteados]).reshape(len(perguntas_votacao), N_TENTATIVAS)
respostas_certas = np.array([f"{a + b:04d}" for a, b in perguntas_votacao])
print(f"{len(repetidas)} respostas sorteadas em {time.perf_counter() - inicio:.0f} s")

numeros_de_tentativas = [1, 3, 5, 7, 11, 15, 21]
acerto_votacao = [float(np.mean([votar(linha[:n]) == certa for linha, certa in zip(tentativas, respostas_certas)]))
                  for n in numeros_de_tentativas]
acerto_guloso = taxa_de_acerto(responder(modelo_meio, perguntas_votacao, resposta_com_pensamento), perguntas_votacao)
for n, acerto in zip(numeros_de_tentativas, acerto_votacao):
    print(f"{n:2d} tentativa(s): acerto {acerto:.1%}")
print(f"resposta gulosa (uma tentativa, sempre o token mais provável): {acerto_guloso:.1%}")

# %%
fig, eixo = plt.subplots(figsize=(8, 4.5))
eixo.plot(numeros_de_tentativas, acerto_votacao, marker="o", label="voto da maioria entre N tentativas")
eixo.axhline(acerto_guloso, color="gray", linestyle="--", label="uma resposta gulosa")
eixo.set_xlabel("número de tentativas por pergunta (o custo cresce na mesma proporção)")
eixo.set_ylabel("taxa de acerto")
eixo.set_title("Mais computação na hora de responder")
eixo.set_xticks(numeros_de_tentativas)
eixo.legend()
eixo.grid(True)
plt.show()

# %% [markdown]
# O ganho é grande nas primeiras tentativas e depois se achata: de 11 para 21 tentativas, o custo quase dobra e o acerto sobe só um pouco.
# Faz sentido: quando a maioria já está certa, mais votos não mudam o resultado; e nas perguntas em que o modelo erra quase sempre do
# mesmo jeito, votar não salva. Neste modelo minúsculo, a votação acaba perto do nível da resposta gulosa, porque os erros dele são
# "escorregões" que o voto e a escolha do token mais provável corrigem de forma parecida. Em modelos grandes, que chegam à resposta por
# caminhos de raciocínio realmente diferentes, a autoconsistência costuma superar a resposta única com folga.
#
# **Computação na hora da resposta.** Até alguns anos atrás, a principal forma de melhorar um modelo era gastar mais computação no
# **treino**: mais dados, mais parâmetros. A autoconsistência mostra outra alavanca: gastar mais computação na **resposta** (test-time compute).
#
# O seletor de esforço do Claude é outra forma da mesma ideia. Em vez de várias tentativas votadas, o modelo pensa por mais tempo: escreve um
# raciocínio mais longo antes de responder. Mais esforço significa mais tokens de pensamento, ou seja, mais tempo e mais custo. E, como no
# gráfico, o benefício não cresce na mesma proporção: em perguntas fáceis, pensar mais quase não muda nada; em problemas difíceis, ajuda
# bastante, mas cada nível a mais rende menos que o anterior. Por isso o Máx custa mais e nem sempre melhora na mesma medida.

# %% [markdown]
# ## 6. Aprender com recompensa
#
# Nas seções anteriores, o modelo aprendeu vendo as respostas certas (aprendizado supervisionado). E se só pudermos dizer "acertou" ou "errou"?
# É o aprendizado por reforço (reinforcement learning, RL): o modelo tenta, recebe uma recompensa (reward) e é ajustado para repetir o que
# deu certo. Quando a recompensa é calculada por um programa que confere a resposta, como o nosso, falamos em aprendizado por reforço com
# recompensa verificável (reinforcement learning with verifiable rewards, RLVR).
#
# É, em essência, o método usado para treinar os modelos de raciocínio: eles resolvem muitos problemas cuja resposta pode ser conferida
# automaticamente (contas, exercícios de matemática, programas com testes) e são reforçados quando acertam. O DeepSeek-R1, por exemplo,
# descreveu publicamente esse tipo de treino em 2025. O truque: ninguém precisa escrever o raciocínio certo; basta saber conferir o resultado.
#
# ### O menor exemplo possível
#
# Um "modelo" que responde 2 + 2 escolhendo entre três respostas, 3, 4 ou 5, com probabilidades dadas pela softmax de três números
# $z = (z_3, z_4, z_5)$, a sua **política** (policy). A regra de ajuste, chamada REINFORCE, é:
#
# $$ z \leftarrow z + \eta \cdot \underbrace{(R - b)}_{\text{vantagem}} \cdot (e_k - p) $$
#
# - $k$ é a resposta sorteada e $R$ a recompensa (1 se acertou, 0 se errou);
# - $b$ é uma linha de base (baseline): a recompensa que o modelo já esperava ganhar. A vantagem (advantage) $R - b$ diz se o resultado foi
#   melhor ou pior que o esperado;
# - $e_k - p$ é o vetor one-hot da resposta sorteada menos as probabilidades: é o gradiente de $\log p_k$, a mesma conta "$p - y$" do Módulo 05,
#   com o sinal trocado.
#
# **Exemplo à mão**, com $z = (0, 0, 0)$, ou seja, $p = (1/3,\ 1/3,\ 1/3)$, $\eta = 1$ e linha de base $b = 1/3$ (a chance atual de acertar):
#
# - **sorteou 4 (acertou):** $R - b = 2/3$ e $e_k - p = (-1/3,\ 2/3,\ -1/3)$. Então $z = (-2/9,\ 4/9,\ -2/9)$, e a probabilidade do 4 sobe para
#   $e^{4/9} / (e^{4/9} + 2e^{-2/9}) \approx 1{,}560 / 3{,}161 \approx 0{,}494$;
# - **sorteou 3 (errou):** $R - b = -1/3$ e $e_k - p = (2/3,\ -1/3,\ -1/3)$. Então $z = (-2/9,\ 1/9,\ 1/9)$: o 3 perde probabilidade, e as outras
#   respostas ganham.

# %%
def softmax_numpy(z):
    """Softmax de um vetor (Módulo 05)."""
    exponenciais = np.exp(z - z.max())
    return exponenciais / exponenciais.sum()


opcoes = [3, 4, 5]
for sorteada in [4, 3]:
    z = np.zeros(3)
    p = softmax_numpy(z)
    k = opcoes.index(sorteada)
    recompensa = 1.0 if sorteada == 4 else 0.0
    vantagem = recompensa - p[1]
    z = z + 1.0 * vantagem * (np.eye(3)[k] - p)
    print(f"sorteou {sorteada}: z = {np.round(z, 3)}   p = {np.round(softmax_numpy(z), 3)}")

# %% [markdown]
# Agora, várias tentativas seguidas. A probabilidade de acertar sobe até perto de 1, sem que ninguém diga ao modelo qual é a resposta certa:
# ele só fica sabendo se acertou.

# %%
gerador_bandido = np.random.default_rng(0)
z = np.zeros(3)
probabilidade_do_4 = []
for tentativa in range(40):
    p = softmax_numpy(z)
    probabilidade_do_4.append(p[1])
    k = gerador_bandido.choice(3, p=p)                  # tenta
    recompensa = 1.0 if opcoes[k] == 4 else 0.0         # confere
    z = z + 1.0 * (recompensa - p[1]) * (np.eye(3)[k] - p)   # ajusta

plt.figure(figsize=(7, 3.5))
plt.plot(probabilidade_do_4, marker=".")
plt.xlabel("tentativa")
plt.ylabel("probabilidade de responder 4")
plt.title("O menor exemplo de aprendizado por reforço")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Agora, com o Transformer
#
# A mesma ideia, aplicada ao `modelo_meio` (o que acerta cerca de dois terços das contas). A cada passo:
#
# 1. **tentar:** sorteamos 16 perguntas novas e, para cada uma, 8 respostas completas, com pensamento, na temperatura 1;
# 2. **conferir:** recompensa 1 se a resposta final estiver certa, 0 se não. Só a resposta final é conferida, nunca os passos;
# 3. **ajustar:** a linha de base de cada resposta é a média das recompensas das 8 tentativas da mesma pergunta, e a vantagem é dividida pelo
#    desvio padrão dessas recompensas. Esse esquema "em grupo" é uma versão simplificada do GRPO, o método do DeepSeek-R1. Em vez de $\log p_k$,
#    usamos a soma dos log-probabilidades de todos os tokens que o modelo escreveu.
#
# A taxa de aprendizado é pequena e cai até zero no fim: no RL, passos grandes desestabilizam o modelo com facilidade. Nos primeiros testes
# deste módulo, com uma taxa maior, o acerto **caía** em vez de subir. São 200 passos, por isso o resultado vem pronto do arquivo de pesos
# (a não ser que você tenha escolhido treinar do zero).

# %%
def treinar_com_recompensa(modelo, passos=200, perguntas_por_passo=16, tentativas_por_pergunta=8, taxa=1e-4,
                           semente=7, avaliar_a_cada=25):
    """RL com recompensa verificável: tenta, confere a resposta final e reforça as tentativas melhores que a média do grupo."""
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    agenda = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda passo: 1 - passo / passos)
    gerador_perguntas = np.random.default_rng(semente)
    gerador_sorteio = torch.Generator().manual_seed(semente)
    tamanho_pergunta, tamanho_resposta = len(pergunta(0, 0)), len(resposta_com_pensamento(0, 0))
    historico = {"passo": [], "acerto_validacao": [], "recompensa_media": []}
    recompensas_recentes = []
    for passo in range(passos + 1):
        if passo % avaliar_a_cada == 0:
            textos = responder(modelo, pares_validacao, resposta_com_pensamento)
            historico["passo"].append(passo)
            historico["acerto_validacao"].append(taxa_de_acerto(textos, pares_validacao))
            historico["recompensa_media"].append(float(np.mean(recompensas_recentes)) if recompensas_recentes else float("nan"))
            recompensas_recentes = []
        if passo == passos:
            break
        # 1. tentar
        perguntas = sortear_pares(perguntas_por_passo, gerador_perguntas, excluir=fora_do_treino)
        repetidas = [par for par in perguntas for _ in range(tentativas_por_pergunta)]
        prefixos = torch.tensor([codificar(pergunta(a, b)) for a, b in repetidas])
        sequencias = gerar(modelo, prefixos, tamanho_resposta, temperatura=1.0, gerador=gerador_sorteio)
        # 2. conferir
        textos = [decodificar(s[tamanho_pergunta:]) for s in sequencias]
        recompensas = torch.tensor([1.0 if extrair_resposta(texto) == f"{a + b:04d}" else 0.0
                                    for texto, (a, b) in zip(textos, repetidas)]).view(perguntas_por_passo, -1)
        recompensas_recentes.append(recompensas.mean().item())
        # 3. ajustar: vantagem em relação à média do grupo, dividida pelo desvio do grupo
        vantagens = (recompensas - recompensas.mean(dim=1, keepdim=True)) / (recompensas.std(dim=1, keepdim=True) + 1e-4)
        log_probabilidades = torch.log_softmax(modelo(sequencias[:, :-1]), dim=-1)
        log_prob_tokens = log_probabilidades.gather(2, sequencias[:, 1:, None])[:, :, 0]
        log_prob_respostas = log_prob_tokens[:, tamanho_pergunta - 1:].sum(dim=1)
        perda = -(vantagens.reshape(-1) * log_prob_respostas).mean()
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        agenda.step()
    return historico


def salvar_pesos(conteudo):
    """Salva os pesos em dados/ (se existir) ou na pasta atual."""
    caminho = CAMINHO_PESOS if os.path.isdir(os.path.dirname(CAMINHO_PESOS)) else NOME_PESOS
    torch.save(conteudo, caminho)
    print("pesos salvos em", caminho)


modelo_recompensa = modelo_com_pesos(pesos_meio)     # começa igual ao modelo do meio do treino
inicio = time.perf_counter()
if TREINAR_DO_ZERO:
    historico_recompensa = treinar_com_recompensa(modelo_recompensa)
    salvar_pesos({
        "direto": modelo_direto.state_dict(), "historico_direto": historico_direto,
        "pensamento": modelo_pensamento.state_dict(), "historico_pensamento": historico_pensamento,
        "pensamento_meio": pesos_meio, "passo_meio": passo_meio,
        "recompensa": modelo_recompensa.state_dict(), "historico_recompensa": historico_recompensa,
    })
else:
    modelo_recompensa.load_state_dict(pesos["recompensa"])
    historico_recompensa = pesos["historico_recompensa"]
print(f"pronto em {time.perf_counter() - inicio:.0f} s")

# %% [markdown]
# **O aprendizado acontecendo**, só com recompensas.

# %%
fig, (eixo_acerto, eixo_recompensa) = plt.subplots(1, 2, figsize=(13, 4))
eixo_acerto.plot(historico_recompensa["passo"], historico_recompensa["acerto_validacao"], marker="o")
eixo_acerto.set_xlabel("passo de aprendizado por reforço")
eixo_acerto.set_ylabel("acerto na validação (resposta gulosa)")
eixo_acerto.set_title("O acerto subindo")
eixo_acerto.grid(True)
eixo_recompensa.plot(historico_recompensa["passo"][1:], historico_recompensa["recompensa_media"][1:], marker="o", color="tab:green")
eixo_recompensa.set_xlabel("passo de aprendizado por reforço")
eixo_recompensa.set_ylabel("recompensa média das tentativas")
eixo_recompensa.set_title("A recompensa subindo")
eixo_recompensa.grid(True)
plt.show()

textos_meio = responder(modelo_meio, pares_teste, resposta_com_pensamento)
textos_recompensa = responder(modelo_recompensa, pares_teste, resposta_com_pensamento)
print(f"acerto no teste: antes do RL {taxa_de_acerto(textos_meio, pares_teste):.1%}   "
      f"depois do RL {taxa_de_acerto(textos_recompensa, pares_teste):.1%}")

# %% [markdown]
# O modelo melhorou sem ver nenhuma resposta certa: só reforçando as próprias tentativas que deram certo, mais do que a média do grupo.
# (A recompensa média fica abaixo do acerto na validação porque, no treino, as tentativas são sorteadas com temperatura 1, mais arriscadas
# que a resposta gulosa.)
# Repare que a recompensa olhava só a resposta final, e mesmo assim os **passos** do raciocínio ficaram mais confiáveis, porque passos
# errados levam a respostas erradas.
#
# Dois cuidados. Primeiro, quando temos o raciocínio certo para mostrar, o treino supervisionado é bem mais eficiente: o modelo com pensamento
# da seção 4 chegou perto de 100% treinando um pouco mais. O RL brilha quando só sabemos **conferir** o resultado. Segundo, o modelo busca a
# recompensa, não a intenção por trás dela: se o verificador tiver uma falha, o modelo pode aprender a explorá-la em vez de resolver o problema
# (em inglês, reward hacking). Por isso, bons verificadores são uma parte central desse tipo de treino.

# %% [markdown]
# ## 7. Do pensar ao agir: um agente com calculadora
#
# Para que fazer a conta "de cabeça", se existe uma calculadora? Um agente (agent) é um modelo que, além de escrever, pode **agir**: chamar
# ferramentas, receber os resultados e continuar. O ciclo tem três partes, repetidas quantas vezes for preciso:
#
# 1. **pensar:** decidir o que fazer;
# 2. **agir:** chamar uma ferramenta (tool call);
# 3. **observar:** ler o resultado (observation) e voltar ao passo 1, ou responder.
#
# Esse padrão foi descrito em 2022 com o nome ReAct (Yao e colegas, de *reasoning* + *acting*). Vamos treinar um modelo para escrever uma chamada
# de ferramenta, `CALC(347+589)`. Quando ele fecha o parêntese, **o nosso programa** (não o modelo) roda a calculadora em Python e acrescenta o
# resultado entre colchetes. Depois, o modelo continua e escreve a resposta:
#
# `347+589=CALC(347+589)[0936]0936.`
#
# No treino, o trecho entre colchetes recebe alvo −100: o modelo não aprende a inventar o resultado da ferramenta, só a usá-lo.
# E usamos as **mesmas 1000 contas** de treino da seção 4.

# %%
def resposta_com_ferramenta(a, b):
    """Chamada da ferramenta, resultado (que no uso real vem da calculadora) e resposta final."""
    return f"CALC({a:03d}+{b:03d})" + f"[{a + b:04d}]" + resposta_direta(a, b)


def montar_exemplos_ferramenta(pares):
    """Como montar_exemplos, mas o trecho entre colchetes (escrito pela ferramenta) também é ignorado no alvo."""
    entradas, alvos = montar_exemplos(pares, resposta_com_ferramenta)
    texto = pergunta(0, 0) + resposta_com_ferramenta(0, 0)
    abre, fecha = texto.index("["), texto.index("]")
    alvos[:, abre - 1:fecha] = -100
    return entradas, alvos


torch.manual_seed(0)
modelo_agente = criar_modelo()
entradas_agente, alvos_agente = montar_exemplos_ferramenta(pares_treino)
otimizador = torch.optim.AdamW(modelo_agente.parameters(), lr=1e-3, weight_decay=0.01)
funcao_perda = nn.CrossEntropyLoss(ignore_index=-100)
gerador_lotes = torch.Generator().manual_seed(0)
inicio = time.perf_counter()
for passo in range(300):
    lote = torch.randint(0, len(entradas_agente), (64,), generator=gerador_lotes)
    perda = funcao_perda(modelo_agente(entradas_agente[lote]).reshape(-1, TAMANHO_VOCABULARIO), alvos_agente[lote].reshape(-1))
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()
print(f"treino do agente em {time.perf_counter() - inicio:.0f} s")

# %% [markdown]
# A ferramenta e o ciclo do agente. A calculadora aceita só o formato "número+número". Não usamos `eval`, que executaria qualquer código:
# o texto escrito por um modelo nunca deve ser tratado como confiável. O laço tem um limite de tamanho, para o caso de o modelo nunca fechar
# o parêntese.

# %%
def calculadora(expressao):
    """A ferramenta: soma dois números escritos como '347+589'."""
    parcela_1, parcela_2 = expressao.split("+")
    return f"{int(parcela_1) + int(parcela_2):04d}"


def chamar_ferramenta(texto, ferramenta):
    """Lê a expressão dentro de CALC(...) e roda a ferramenta; qualquer problema vira '0000' (falha)."""
    try:
        resultado = ferramenta(texto[texto.index("CALC(") + 5:texto.index(")")])
    except ValueError:
        return "0000"
    return resultado if len(resultado) == 4 and resultado.isdigit() else "0000"


def proximo_caractere(modelo, texto):
    """O caractere mais provável depois do texto."""
    return decodificar(gerar(modelo, torch.tensor([codificar(texto)]), 1)[0, -1:])


def rodar_agente(modelo, a, b, ferramenta=calculadora, mostrar=False):
    """Pensar e agir (escrever a chamada), observar (rodar a ferramenta) e responder."""
    texto = pergunta(a, b)
    while not texto.endswith(")") and len(texto) < 30:          # pensar e agir
        texto += proximo_caractere(modelo, texto)
    resultado = chamar_ferramenta(texto, ferramenta)              # a ferramenta roda fora do modelo
    texto += "[" + resultado + "]"                                # observar
    while not texto.endswith(".") and len(texto) < 45:          # responder
        texto += proximo_caractere(modelo, texto)
    if mostrar:
        print("pergunta:              ", pergunta(a, b))
        print("pensar e agir (modelo):", texto[len(pergunta(a, b)):texto.index("[")])
        print("observar (ferramenta): ", resultado)
        print("responder (modelo):    ", texto[texto.index("]") + 1:])
    return texto


rodar_agente(modelo_agente, 347, 589, mostrar=True);

# %% [markdown]
# (O `;` no fim da linha só evita que o Jupyter imprima o texto devolvido pela função.)
#
# Agora, as 500 contas de teste. Para ir mais rápido, a versão abaixo faz o mesmo ciclo para todas as perguntas de uma vez.

# %%
def avaliar_agente(modelo, pares, ferramenta=calculadora):
    """O ciclo do agente para muitas perguntas ao mesmo tempo; devolve a taxa de acerto."""
    prefixos = torch.tensor([codificar(pergunta(a, b)) for a, b in pares])
    tamanho_chamada = len("CALC(000+000)")
    sequencias = gerar(modelo, prefixos, tamanho_chamada)               # pensar e agir
    observacoes = ["[" + chamar_ferramenta(decodificar(s), ferramenta) + "]" for s in sequencias]
    sequencias = torch.cat([sequencias, torch.tensor([codificar(o) for o in observacoes])], dim=1)   # observar
    sequencias = gerar(modelo, sequencias, len(resposta_direta(0, 0)))  # responder
    return taxa_de_acerto([decodificar(s) for s in sequencias], pares)


print(f"acerto do agente no teste: {avaliar_agente(modelo_agente, pares_teste):.1%}")

# %% [markdown]
# Com as mesmas 1000 contas que fizeram o modelo direto apenas decorar, o agente acerta praticamente tudo, porque não precisou aprender a
# somar: só a pedir a conta e copiar o resultado. Delegar para a ferramenta certa é uma forma poderosa de "raciocinar".
#
# **E se a ferramenta errar?** O agente confia no que observa. Uma calculadora com defeito, que "esquece" o algarismo do milhar:

# %%
def calculadora_com_defeito(expressao):
    """Erra sempre que a soma passa de 999: troca o algarismo do milhar por zero."""
    return "0" + calculadora(expressao)[1:]


print(f"acerto do agente com a calculadora defeituosa: {avaliar_agente(modelo_agente, pares_teste, calculadora_com_defeito):.1%}")

# %% [markdown]
# O agente repete o erro da ferramenta sem desconfiar. Agentes reais precisam de ferramentas confiáveis, de permissões bem definidas (o que
# podem ou não fazer) e, de preferência, de formas de conferir os resultados.
#
# **É assim que o Claude Code trabalha.** Ele é um modelo de linguagem num ciclo de agente: lê o pedido, pensa, chama uma ferramenta (ler um
# arquivo, editar código, rodar um comando no terminal), observa o resultado e decide o próximo passo. Este curso foi criado desse jeito: para
# cada módulo, o Claude Code leu o `CLAUDE.md` e o `PROGRESSO.md`, escreveu o arquivo-fonte, rodou o notebook do início ao fim para testá-lo,
# leu os erros e os números que saíram, corrigiu o que não batia e publicou no GitHub. Neste módulo, por exemplo, foi observando o acerto cair nos
# primeiros testes de aprendizado por reforço que a taxa de aprendizado foi ajustada. A diferença para o nosso agente está na escala: dezenas de
# ferramentas, centenas de ciclos por tarefa e um modelo enorme decidindo o que fazer.

# %% [markdown]
# ## 8. O que está por vir
#
# Ninguém sabe ao certo como a IA vai evoluir. Mas dá para ver em que direções os pesquisadores trabalham, separando o que já funciona do que
# ainda é aposta.
#
# **Agentes que trabalham sozinhos por mais tempo.**
#
# - *Já funciona:* agentes que programam, pesquisam e executam tarefas de minutos a algumas horas, com uma pessoa acompanhando e revisando.
# - *Aposta:* agentes que trabalham dias ou semanas sem supervisão, de forma confiável. Uma medição de 2025 (do grupo de pesquisa METR) indica que a
#   duração das tarefas que os agentes conseguem completar vem dobrando a cada cerca de sete meses; ninguém sabe por quanto tempo essa tendência
#   continua. Erros que se acumulam ao longo de muitos passos, como no pêndulo do Módulo B1, são um dos grandes obstáculos.
#
# **Raciocínio mais forte.**
#
# - *Já funciona:* com cadeia de pensamento, mais computação na resposta e aprendizado por reforço com recompensas verificáveis, os modelos
#   melhoraram muito em matemática e programação. Em 2025, sistemas de IA alcançaram pontuação de medalha de ouro na Olimpíada Internacional
#   de Matemática.
# - *Aposta:* raciocínio confiável em problemas abertos, em que não há como conferir a resposta automaticamente, e descobertas científicas
#   originais com frequência.
#
# **Robôs e modelos de mundo.**
#
# - *Já funciona:* robôs que aprendem tarefas de manipulação a partir de demonstrações, em ambientes controlados; modelos que geram vídeos e
#   simulações interativas.
# - *Aposta:* robôs úteis em casas e em lugares imprevisíveis, e modelos de mundo (world models) que prevejam as consequências das ações com
#   precisão suficiente para planejar. É o problema do Módulo B1 em escala gigante: aprender a dinâmica do mundo sem que os erros se acumulem.
#
# **Entender o que acontece dentro dos modelos e torná-los seguros.**
#
# - *Já funciona:* a interpretabilidade (interpretability) já identifica alguns conceitos e "circuitos" dentro de modelos grandes, como fizemos
#   em miniatura com os mapas de atenção do Módulo 10. Também existem avaliações sistemáticas de capacidades perigosas e técnicas para deixar os
#   modelos mais honestos e mais difíceis de usar para o mal.
# - *Aposta:* entender os modelos a ponto de garantir que se comportem como queremos, mesmo quando forem mais capazes que nós em muitas tarefas.
#   Hoje, entendemos só uma pequena parte do que acontece dentro deles.
#
# O que muda pouco são as ideias deste curso: uma função de perda, gradientes calculados pela retropropagação, um otimizador dando passos,
# atenção, probabilidades e, agora, recompensas. Os modelos da fronteira são construídos com essas mesmas peças.

# %% [markdown]
# ## Mão na massa
#
# Crie células novas logo abaixo para cada exercício.
#
# **Exercício 1 (fácil).** Escreva à mão o texto com pensamento da conta 478 + 256, no formato do módulo, e confira com
# `resposta_com_pensamento(478, 256)`. Depois, ache à mão a resposta mais votada entre 0734, 0724, 0734, 0634 e 0734, e confira com `votar`.
#
# **Exercício 2 (médio).** Repita a autoconsistência da seção 5 com 11 tentativas por pergunta, nas primeiras 100 perguntas, para as temperaturas
# 0,3, 0,7 e 1,2. Mostre, para cada temperatura, o acerto de uma tentativa só e o acerto da votação. O que acontece com temperatura baixa demais?
# E alta demais?
#
# **Exercício 3 (desafio).** Às vezes temos um **verificador**: um programa que confere uma resposta. Com as tentativas sorteadas na seção 5
# (`tentativas`), escolha, para cada pergunta, a primeira tentativa aprovada pelo verificador (se nenhuma for aprovada, use a votação).
# O verificador confere com a operação inversa: a resposta menos a primeira parcela deve dar a segunda. Faça o gráfico do acerto contra o
# número de tentativas para a votação e para o verificador. O que a diferença entre as curvas diz sobre o valor de poder conferir respostas?

# %% [markdown]
# ## Vocabulário em inglês
#
# | Inglês | Português | Exemplo em inglês |
# |---|---|---|
# | chain of thought | cadeia de pensamento | *Writing a chain of thought helps the model add numbers.* |
# | reasoning model | modelo de raciocínio | *A reasoning model thinks before it answers.* |
# | scratchpad | rascunho | *The written steps work as a scratchpad.* |
# | memorization | memorização (decorar) | *With only 1000 examples, the direct model fell back on memorization.* |
# | greedy decoding | decodificação gulosa | *Greedy decoding always picks the most likely token.* |
# | self-consistency | autoconsistência | *Self-consistency samples many answers and takes a vote.* |
# | majority vote | voto da maioria | *The majority vote picked the right answer.* |
# | test-time compute | computação na hora da resposta | *More test-time compute costs more money and time.* |
# | reinforcement learning | aprendizado por reforço | *Reinforcement learning learns from rewards.* |
# | reward | recompensa | *The reward is one for a correct answer.* |
# | verifiable reward | recompensa verificável | *A program checks the answer to give a verifiable reward.* |
# | policy | política | *The policy gives the probability of each answer.* |
# | advantage | vantagem | *A positive advantage means better than expected.* |
# | agent | agente | *An agent thinks, acts and observes.* |
# | tool call | chamada de ferramenta | *The agent made a tool call to the calculator.* |

# %% [markdown]
# ## Resumo
#
# - Escrever os passos (cadeia de pensamento) quebra um problema difícil em vários pequenos e dá ao modelo mais computação: com só 1000 contas,
#   o modelo que pensa aprendeu o algoritmo, e o direto decorou o treino e errou quase metade das contas novas.
# - Gastar mais computação na resposta, com várias tentativas e voto da maioria ou pensando por mais tempo, melhora o acerto, mas os ganhos
#   diminuem enquanto o custo cresce.
# - No aprendizado por reforço com recompensa verificável, o modelo só sabe se acertou; reforçar as próprias tentativas melhores que a média
#   já melhora o raciocínio, e é assim que os modelos de raciocínio são treinados, em escala gigante.
# - Um agente alterna pensar, agir (chamar ferramentas) e observar; ele confia nas ferramentas, que por isso precisam ser confiáveis e seguras.
# - Agentes mais autônomos, raciocínio mais forte, robôs e modelos de mundo, interpretabilidade e segurança: há resultados sólidos e muitas
#   apostas em aberto.
#
# **Fim do curso.** Você começou com variáveis e uma pedra caindo e terminou treinando, à mão e em miniatura, as ideias que movem os modelos
# mais avançados. Parabéns pela disciplina, e bons estudos daqui em diante.

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
# Unidades: $8 + 6 + 0 = 14$ (escreve 4, vai 1); dezenas: $7 + 5 + 1 = 13$ (escreve 3, vai 1); centenas: $4 + 2 + 1 = 07$ (escreve 7, vai 0).
# Texto: `8+6+0=14,7+5+1=13,4+2+1=07,0734.` A mais votada é 0734, com 3 votos.

# %%
print(resposta_com_pensamento(478, 256))
print(votar(["0734", "0724", "0734", "0634", "0734"]))

# %% [markdown]
# ### Solução do Exercício 2

# %%
perguntas_ex2 = pares_teste[:100]
certas_ex2 = np.array([f"{a + b:04d}" for a, b in perguntas_ex2])
for temperatura in [0.3, 0.7, 1.2]:
    repetidas_ex2 = [par for par in perguntas_ex2 for _ in range(11)]
    textos_ex2 = responder(modelo_meio, repetidas_ex2, resposta_com_pensamento, temperatura=temperatura,
                           gerador=torch.Generator().manual_seed(1))
    tentativas_ex2 = np.array([extrair_resposta(texto) for texto in textos_ex2]).reshape(len(perguntas_ex2), 11)
    uma = float(np.mean(tentativas_ex2[:, 0] == certas_ex2))
    votacao = float(np.mean([votar(linha) == certa for linha, certa in zip(tentativas_ex2, certas_ex2)]))
    print(f"temperatura {temperatura}: uma tentativa {uma:.1%}   votação de 11 {votacao:.1%}")

# %% [markdown]
# Com temperatura baixa, as tentativas quase não variam: cada uma acerta bastante, mas a votação ganha pouco, porque todas repetem o mesmo
# raciocínio, inclusive os mesmos erros. Com temperatura alta, as tentativas ficam variadas, mas cada uma erra tanto que nem a votação
# compensa. Neste modelo minúsculo, a temperatura baixa acabou vencendo: os erros dele são escorregões, e sortear pouco já basta. Em modelos
# grandes, em que caminhos de raciocínio diferentes realmente ajudam, a autoconsistência costuma usar temperaturas intermediárias, para
# equilibrar diversidade e qualidade das tentativas.

# %% [markdown]
# ### Solução do Exercício 3

# %%
def verificar(resposta, a, b):
    """Confere com a operação inversa: resposta - a deve dar b."""
    return resposta.isdigit() and int(resposta) - a == b


acerto_verificador = []
for n in numeros_de_tentativas:
    escolhidas = []
    for linha, (a, b) in zip(tentativas, perguntas_votacao):
        aprovadas = [resposta for resposta in linha[:n] if verificar(resposta, a, b)]
        escolhidas.append(aprovadas[0] if aprovadas else votar(linha[:n]))
    acerto_verificador.append(float(np.mean(np.array(escolhidas) == respostas_certas)))

fig, eixo = plt.subplots(figsize=(8, 4.5))
eixo.plot(numeros_de_tentativas, acerto_votacao, marker="o", label="voto da maioria")
eixo.plot(numeros_de_tentativas, acerto_verificador, marker="s", label="primeira tentativa aprovada pelo verificador")
eixo.set_xlabel("número de tentativas por pergunta")
eixo.set_ylabel("taxa de acerto")
eixo.set_xticks(numeros_de_tentativas)
eixo.legend()
eixo.grid(True)
plt.show()
for n, votacao, verificada in zip(numeros_de_tentativas, acerto_votacao, acerto_verificador):
    print(f"{n:2d} tentativas: votação {votacao:.1%}   verificador {verificada:.1%}")

# %% [markdown]
# Com um verificador, basta que **uma** das tentativas esteja certa, e o acerto sobe bem mais que com a votação, chegando perto de 100%.
# A diferença mostra quanto "conhecimento escondido" o modelo tem: ele quase sempre consegue chegar à resposta certa em alguma tentativa,
# só não sabe qual é. É por isso que poder conferir respostas vale tanto: permite escolher a tentativa certa na hora (como fazem agentes que
# rodam os testes de um programa) e, no treino, permite o aprendizado por reforço com recompensa verificável, que transforma esses acertos
# ocasionais em acertos de primeira.
