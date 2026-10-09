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

## Módulo 04: Primeira rede neural (2026-10-09)
- Arquivos: `fontes/04-primeira-rede-neural.py`, `modulos/04-primeira-rede-neural.ipynb`.
- Conceitos: XOR como lâmpada de escada (three-way), linearmente separável, neurônio, camada escondida/saída, rede XOR montada à mão (OU, E) e o espaço escondido, por que a ativação é necessária, sigmoide × ReLU, regra da cadeia, retropropagação à mão (rede 1-1-1: x=1, w1=0.5, w2=1, y=1 → dL/dw1 = −0.0821), tabela elo → matriz, inicialização aleatória (quebra de simetria), dicionário Python, gradient checking com `np.ndindex`, gradiente que encolhe na sigmoide (≤ 0.25).
- Funções: `relu`, `derivada_relu`, `inicializar_rede(n_entradas, n_ocultos, semente)` → dict W1,b1,W2,b2 (N(0,1)), `propagacao_frente(X, rede)` → P, memoria, `retropropagacao(y, rede, memoria)`, `conferir_gradientes`, `treinar_rede(X, y, n_ocultos, taxa, passos, semente)`, `acuracia_rede`, `desenhar_regioes(eixo, calcular_probabilidade, X, y, titulo)`. y é coluna (n, 1).
- Dados: nuvem XOR, 50 pontos por canto, ruído 0.15, semente 42. Regressão logística 52%; rede 8 ReLU, taxa 0.5, 3000 passos: 100%.
- Para o Módulo 05: generalizar a saída para softmax (10 classes) e treinar com minilotes; a rede atual tem uma saída sigmoide.

## Módulo 05: Dígitos (2026-10-09)
- Arquivos: `fontes/05-digitos.py`, `modulos/05-digitos.ipynb`.
- Conceitos: pixel, imagem como tabela, achatar (64), X/16, generalização, treino/validação/teste 60/20/20 (permutação, semente 42), softmax (estável, subtrai o máximo), one-hot, entropia cruzada multiclasse, gradiente p − y conferido numericamente, inicialização de He, minilotes, época, SGD, sobreajuste (100 imagens: 100% treino × ~89% validação), parada antecipada, matriz de confusão 10×10, galeria de erros.
- Funções: `softmax(Z)`, `codificar_one_hot(rotulos, n_classes=10)`, `entropia_cruzada(P, Y)`, `inicializar_rede(n_entradas, n_ocultos, n_saidas, semente)` (He), `propagacao_frente`, `retropropagacao`, `conferir_gradientes`, `avaliar(X, Y, rede)` → perda, acurácia, `treinar_minilotes(X, Y, X_val, Y_val, n_ocultos, taxa, epocas, tamanho_lote, semente)` → rede, historico (perda/acc treino/val), `desenhar_historico`.
- Resultado: 64-64-10 ReLU, taxa 0.1, lote 32, 40 épocas → treino 99.4%, validação 97.5%, teste 95.6% (16 erros em 360).
- Para o Módulo 07: refazer esta rede em PyTorch e comparar (mesma divisão de dados e semente).

## Módulo 06: Física + IA, lançamento oblíquo (2026-10-09)
- Arquivos: `fontes/06-fisica-lancamento.py`, `modulos/06-fisica-lancamento.ipynb`.
- Conceitos: R = v0² sen(2θ)/g (dedução e exemplo à mão), arrasto quadrático a = −c|v|v (c = 0.013 m⁻¹, bola de futebol), método de Euler (exemplo à mão, primeira ordem), interpolação linear no pouso, ângulo ótimo < 45° com ar, regressão com rede (saída linear + MSE, dZ2 = 2(ŷ − y)/n), normalizar alvo, estatísticas só do treino, erro médio absoluto, modelo substituto, extrapolação, quando usar IA em Física.
- Funções: `alcance_sem_ar(v0, angulo_graus)`, `simular_trajetoria(v0, angulo_graus, c, dt)` → x, y, `simular_alcances(velocidades, angulos_graus, c, dt)` (vetorizado), `inicializar_rede`/`propagacao_frente`/`retropropagacao` (regressão), `conferir_gradientes`, `treinar(X, y, X_val, y_val, n_ocultos, taxa, epocas, tamanho_lote, semente)`, `prever_alcance(rede, v, ângulo)`.
- Dados: 2000 chutes, v U(10,35), θ U(10,70), semente 42, divisão 1200/400/400. Rede 2-64-1, taxa 0.05, 300 épocas: erro médio 0.34 m no teste; fora da faixa (60 m/s, 40°) prevê 86 m contra 99 m simulados. Desafio: atributo "alcance sem ar" reduz o erro para 0.17 m, mas extrapola para cima.
- Observação: neste contêiner o NumPy é lento com arrays grandes; não prometer razões fixas de tempo nos textos.
- Para o Módulo 07: refazer em PyTorch a rede do Módulo 05 (softmax, dígitos) e conferir o autograd com o exemplo à mão do Módulo 04 (x=1, w1=0.5, w2=1, y=1 → dL/dw1 = −0.0821, dL/dw2 = −0.2174).

## Módulo 07: PyTorch (2026-10-09)
- Arquivos: `fontes/07-pytorch.py`, `modulos/07-pytorch.ipynb`.
- Conceitos: tensor, float32 × float64, conversão NumPy ↔ PyTorch, `requires_grad`, `.backward()`, `.grad`, grafo computacional, gradientes do exemplo do Módulo 04 conferidos, acúmulo de gradientes (zerar), `torch.no_grad()`, otimizadores (SGD, momento como inércia, Adam), `nn.Linear`/`nn.ReLU`/`nn.Sequential`, peso (saídas, entradas), `nn.CrossEntropyLoss` com logits e rótulos inteiros, gradientes PyTorch = NumPy do Módulo 05 (diferença ~1e-17), escolha pela validação, tabela NumPy × PyTorch.
- Funções: `treinar_reta(criar_otimizador, passos)`, `criar_rede_digitos()`, `para_tensores(indices)`, `acuracia(rede, X, y)`, `treinar_digitos(rede, otimizador, epocas=40, tamanho_lote=32, semente=0)` (usa X_treino/y_treino globais), `contar_parametros`.
- Resultados: vale do Módulo 01 (200 passos): momento chega ao mínimo, Adam (taxa 1) quase; dígitos: SGD 0.1 → val 97.5%, Adam 0.001 → val 98.3%, teste 96.1%.
- Instalação: o índice download.pytorch.org está bloqueado pelo proxy; `pip install -q torch` (PyPI) funciona, mas é grande e demora alguns minutos.
- Para o Módulo 08: o corpus em inglês vai em `dados/` (100 KB a 1 MB), com download de raw.githubusercontent.com quando o notebook não achar `../dados/`.
