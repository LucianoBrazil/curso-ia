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
