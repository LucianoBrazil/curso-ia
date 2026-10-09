# Progresso do curso

## Configuração (2026-10-09)
- Criados `CLAUDE.md`, `README.md` (índice), `requirements.txt`, pastas `fontes/`, `modulos/`, `dados/`.

## Módulo 00: Ferramentas (2026-10-09)
- Arquivos: `fontes/00-ferramentas.py`, `modulos/00-ferramentas.ipynb`.
- Conceitos: variáveis, tipos, listas, f-strings, funções, `for`, `if`, arrays NumPy, vetorização, `linspace`, `arange`, máscaras booleanas, `np.random.default_rng`, gráficos com `plt.plot`/`plt.scatter`.
- Exemplo central: queda livre, `posicao_queda(t, g=9.8)` = g t²/2; `velocidade_queda(t)` = g t.
- Aluno já viu: `rng = np.random.default_rng(42)`, ruído gaussiano em medições, comparação laço × vetorização com `time`.
- Dados: gerados no notebook.
- Para o Módulo 01: o aluno sabe gerar dados com ruído e plotar pontos + curva; pode partir direto para o erro quadrático médio.

## Módulo 01: Ajustando uma reta (2026-10-09)
- Arquivos: `fontes/01-ajustando-uma-reta.py`, `modulos/01-ajustando-uma-reta.ipynb`.
- Conceitos: modelo, parâmetro/peso, viés, resíduo, função de perda, MSE, paisagem do erro, derivada (passinho e fórmula via quadrado da soma), derivada parcial, gradiente, gradiente descendente, taxa de aprendizado (pequena/boa/grande, limite 1/média(t²)), hiperparâmetro, mínimos quadrados, `np.polyfit`.
- Funções: `prever(tempo, w, b=0.0)`, `erro_quadratico_medio(previsto, medido)`, `derivada_erro(tempo, distancia, w)`, `gradiente_descendente_1d(...)`, `gradiente_descendente(tempo, distancia, w_inicial, b_inicial, taxa, passos)` (devolve históricos de w, b, erro), `minimos_quadrados(tempo, distancia)`.
- Dados: carro a 20 m/s, t = linspace(0, 10, 25), ruído N(0, 10), semente 42; 2º experimento com b = 30 m.
- Gráficos: tangentes na parábola, erro caindo (log), mapa de contorno E(w, b) com o caminho.
- Para o Módulo 02: o vale estreito de E(w, b) (b converge devagar porque t vai até 10) foi anunciado como motivação para normalizar.

## Módulo 02: Várias variáveis (2026-10-09)
- Arquivos: `fontes/02-varias-variaveis.py`, `modulos/02-varias-variaveis.ipynb`.
- Conceitos: vetor, produto escalar (`@`), matriz (linhas = exemplos, colunas = atributos), Xw + b, atributo/alvo, gradiente vetorial (2/n) Xᵀe, transposta, vetorização (laços × `X.T @`), escala, padronização z-score, desnormalizar pesos, `np.linalg.lstsq` com coluna de uns, engenharia de atributos (t²).
- Funções: `prever(X, w, b)`, `erro_quadratico_medio`, `gradientes(X, y, w, b)`, `treinar_regressao(X, y, taxa, passos)` (começa de zeros; devolve w, b, histórico), `normalizar(X)` → (X_norm, media, desvio), `desnormalizar_pesos(w, b, media, desvio)`.
- Dados: 200 dias de consumo de energia (banho min, horas de ar, lavagens; pesos 0.09, 1.0, 0.8; viés 2.5; ruído 0.5; semente 42). Bola lançada: h = 1.5 + 15t − 4.9t², 60 pontos. Desafio: pêndulo T² = 4π²L/g.
- Taxas: sem normalizar diverge a partir de 0.0008; normalizado usa 0.1 a 0.3.
- Para o Módulo 03: o aluno já usa `@`, `X.T`, `normalizar`; pode reaproveitar o padrão de treino com dois atributos (horas de estudo, exercícios).

## Módulo 03: Regressão logística (2026-10-09)
- Arquivos: `fontes/03-regressao-logistica.py`, `modulos/03-regressao-logistica.ipynb`.
- Conceitos: classificação binária, classe, probabilidade, sigmoide (σ' = p(1−p), nota Fermi-Dirac), z = Xw + b, entropia cruzada, por que não MSE (gradiente some com p(1−p)), regra da cadeia apresentada como "engrenagens" (o Módulo 04 aprofunda), gradiente (1/n)Xᵀ(p−y), limiar, fronteira de decisão (reta), acurácia, matriz de confusão (VN, FP, FN, VP), comparação com `LogisticRegression(C=1e6)`.
- Funções: `sigmoide`, `prever_probabilidade(X, w, b)`, `entropia_cruzada(p, y)` (com clip), `gradientes_logistica`, `acuracia`, `treinar_logistica(X, y, taxa, passos, w_inicial=None, b_inicial=0.0)` → w, b, hist_perda, hist_acuracia, `matriz_confusao`, `desenhar_matriz_confusao`, `desenhar_alunos`; cópias de `normalizar`/`desnormalizar_pesos`.
- Dados: 200 alunos, horas U(0,20), exercícios inteiros 0–50, regra z = 0.4h + 0.1e − 7, rótulo sorteado (Bernoulli), semente 42. Acurácia no treino: 80%.
- Avisado: acurácia medida no próprio treino; separação treino/teste fica para o Módulo 05.
- Para o Módulo 04: o "próximo módulo" anunciado é o caso em que uma reta não separa as classes (XOR).
