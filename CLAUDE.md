# Curso: Inteligência Artificial do Zero

Instruções permanentes para o Claude Code. Valem para todas as sessões deste repositório (LucianoBrazil/curso-ia).

## O que é este repositório

Um curso de inteligência artificial em português do Brasil, feito sob medida para um único aluno, em notebooks Jupyter que ele estuda pelo Google Colab. O curso parte do zero e usa Matemática, Física e Inglês, que são os interesses do aluno, como fio condutor.

Cada sessão produz **um módulo**, testa, publica e para.

## Perfil do aluno

- Adulto, autodidata, disciplinado. Quer entender o que acontece por dentro, não só usar ferramentas prontas.
- Gosta de Matemática, Física e Inglês.
- Python: iniciante.
- Matemática: domina o ensino médio. Derivadas, vetores, matrizes e probabilidade devem ser apresentados de forma intuitiva, no momento em que forem necessários, sem pressupor cálculo.
- Também estuda inglês, por isso todo termo técnico aparece nas duas línguas.

## Orçamento (importante)

O curso inteiro conta com cerca de US$ 100 de crédito para 11 módulos, em torno de US$ 9 por módulo. Trabalhe de forma econômica:

- Faça só o módulo pedido. Ao terminar, pare. Não comece o próximo sem pedido.
- Não releia módulos anteriores. Use o `PROGRESSO.md` para saber o que já existe. Se precisar de um detalhe, leia só o trecho necessário do arquivo-fonte `.py`. Nunca abra um `.ipynb` para leitura.
- Instale pacotes com `pip install -q` e não imprima logs longos (use `| tail -n 20` quando precisar ver um erro).
- Teste uma vez; se falhar, corrija e teste de novo. No máximo 3 rodadas de correção. Se o problema persistir, simplifique o código.
- Não use subagentes nem tarefas em paralelo.
- Respostas no chat: curtas e em português.

## Estrutura do repositório

- `README.md`: índice do curso, com links "Abrir no Colab"
- `CLAUDE.md`: este arquivo
- `PROGRESSO.md`: diário curto do que já foi feito (você mantém)
- `requirements.txt`: dependências para rodar fora do Colab
- `fontes/`: fonte de cada módulo em .py (formato percent do Jupytext)
- `modulos/`: notebooks .ipynb gerados a partir de `fontes/`, SEM saídas
- `dados/`: arquivos de dados pequenos, quando necessários

## Primeira sessão (configuração)

Se o `README.md` ainda não tiver o índice do curso, antes do módulo pedido:

1. Crie as pastas, o `requirements.txt` (numpy, matplotlib, scikit-learn, torch) e o `PROGRESSO.md`.
2. Reescreva o `README.md`: título, uma frase sobre o curso, a seção "Como estudar" (abrir pelo link do Colab; executar as células em ordem com Shift+Enter; fazer os exercícios antes de olhar as soluções) e uma tabela com os 11 módulos. Módulos ainda não feitos aparecem sem link.

## Como produzir um módulo

1. Escreva a fonte em `fontes/MM-nome-curto.py`, no formato percent do Jupytext (`# %%` para código, `# %% [markdown]` para texto). Sai mais barato e é mais fácil de editar do que JSON.
2. Gere o notebook, por exemplo: `jupytext --to ipynb --set-kernel python3 fontes/MM-nome-curto.py -o modulos/MM-nome-curto.ipynb`.
3. Teste executando uma cópia do início ao fim, sem alterar o arquivo publicado, por exemplo: `jupyter nbconvert --to notebook --execute modulos/MM-nome-curto.ipynb --output-dir /tmp`. O notebook em `modulos/` fica SEM saídas: o aluno executa no Colab.
4. Atualize o `README.md` (link do módulo) e o `PROGRESSO.md`.
5. Publique (veja "Publicação").

Pacotes para os testes: `pip install -q numpy matplotlib scikit-learn jupytext nbconvert ipykernel`. A partir do Módulo 07, também `torch`: prefira o build de CPU (`pip install -q torch --index-url https://download.pytorch.org/whl/cpu`); se esse endereço estiver bloqueado, instale do PyPI normalmente.

Para ajustes pedidos depois, edite a fonte `.py` e gere o `.ipynb` de novo.

## Padrão de cada notebook

1. Título e o selo "Abrir no Colab":
   `[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/MM-nome-curto.ipynb)`
2. Objetivo do módulo em 2 ou 3 linhas e o que o aluno vai construir.
3. Intuição antes da fórmula: analogia, exemplo numérico pequeno ou gráfico.
4. A Matemática necessária, do zero, em LaTeX nas células de texto, com um exemplo calculado à mão.
5. Código em células curtas, cada uma precedida de explicação. Comentários em português e nomes de variáveis claros.
6. Pelo menos um gráfico que mostre o aprendizado acontecendo (o erro caindo, uma fronteira se formando etc.).
7. "Mão na massa": 3 exercícios graduados (fácil, médio, desafio). As soluções ficam numa seção "Soluções" no fim, depois de um aviso para não espiar antes de tentar.
8. "Vocabulário em inglês": tabela com 8 a 15 termos do módulo (inglês, português e uma frase curta de exemplo em inglês).
9. Resumo em 5 tópicos e uma linha sobre o próximo módulo.

Regras gerais:

- Na primeira menção de um termo técnico, escreva em português com o inglês entre parênteses. Exemplo: gradiente descendente (gradient descent).
- Tudo roda no Colab gratuito, em CPU, sem GPU, chave de API ou conta. Cada treino leva no máximo uns 2 minutos; o notebook inteiro, no máximo uns 5.
- Cada notebook é independente: o Colab abre só o notebook, sem o resto do repositório. Se precisar de uma função de um módulo anterior, copie-a com uma nota do tipo "vimos isso no Módulo 04".
- Fixe sementes aleatórias para os resultados se repetirem.
- Bibliotecas: numpy, matplotlib e scikit-learn (só para datasets prontos e comparações). PyTorch só a partir do Módulo 07.
- Até o Módulo 06, implemente tudo à mão com NumPy. O objetivo é o aluno ver o que acontece por dentro.
- Tom de professor paciente e direto, frases curtas, sem emojis.

## Dados

- Prefira dados gerados no próprio notebook (simulações físicas, dados sintéticos) ou datasets que vêm embutidos no scikit-learn, sem download (por exemplo `load_digits`).
- Para os módulos 08 a 10, use um texto em inglês de domínio público salvo em `dados/` (de 100 KB a 1 MB). O curso usa Alice in Wonderland, do corpus Gutenberg do NLTK (`dados/alice.txt`; no Colab, obtido com `nltk.download("gutenberg")`). Se nenhum download funcionar, escreva você mesmo um corpus de frases simples e variadas em inglês, com pelo menos 100 KB, e diga isso no notebook. Registre a fonte no notebook e no `PROGRESSO.md`. O mesmo arquivo serve para os três módulos.
- O repositório é privado: o Colab não consegue baixar arquivos dele pela URL raw do GitHub, e nenhum notebook pode depender disso. O notebook procura o arquivo em `../dados/` (quem roda com o repositório inteiro) e, quando fizer sentido, na pasta atual (quem enviou o arquivo pelo painel de arquivos do Colab). Se não achar, obtém os dados de uma fonte pública que funcione no Colab (por exemplo, `nltk.download("gutenberg")`, instalando a `nltk` se preciso) ou, no caso de pesos de modelos, treina do zero automaticamente e avisa quanto tempo isso leva. Teste também o caminho sem os arquivos locais, rodando uma cópia do notebook numa pasta sem `../dados/`.

## Plano do curso

**00 — Ferramentas: Python, NumPy e gráficos** (o aluno pode pular se já programar)
Variáveis, listas, funções, laços e condicionais; arrays do NumPy e operações vetorizadas; gráficos com matplotlib. Exemplos com sabor de Física, como posição e tempo de um objeto em queda livre.

**01 — O que é "aprender": ajustando uma reta**
Medições com ruído de um carro em velocidade constante (distância × tempo). Erro quadrático médio; a "paisagem do erro"; derivada como inclinação; gradiente descendente passo a passo; taxa de aprendizado (learning rate) grande ou pequena demais. Comparação com a solução exata por mínimos quadrados.

**02 — Várias variáveis: vetores, matrizes e normalização**
Regressão linear múltipla em notação matricial; produto escalar; vetorização; o gradiente como vetor; por que normalizar os dados; primeira ideia de engenharia de atributos (feature engineering). Dados sintéticos com sentido físico, escolhidos para que um modelo linear faça sentido.

**03 — Classificação: regressão logística**
Sigmoide; probabilidade; entropia cruzada (cross-entropy) e por que não usar o erro quadrático; fronteira de decisão; acurácia e matriz de confusão. Exemplo com dados sintéticos: aprovação numa prova a partir de horas de estudo e de exercícios resolvidos.

**04 — A primeira rede neural, do zero**
Neurônio artificial; camadas; funções de ativação (sigmoide, ReLU); propagação para frente (forward pass); retropropagação (backpropagation) pela regra da cadeia, primeiro num exemplo numérico feito à mão e depois em código; o OU exclusivo (XOR) e por que uma reta não basta; conferência numérica dos gradientes (gradient checking).

**05 — Reconhecendo dígitos escritos à mão**
`load_digits` (imagens 8×8); softmax; treino, validação e teste; minilotes (mini-batches); sobreajuste (overfitting) e como percebê-lo; exibição dos dígitos em que a rede erra.

**06 — Física + IA: o lançamento oblíquo**
Fórmula do alcance sem resistência do ar. Com resistência do ar não há fórmula simples, então simule pelo método de Euler. Gere dados (ângulo e velocidade → alcance), treine a rede dos módulos anteriores (NumPy) para prever o alcance e compare com a simulação. Mostre o que acontece fora da faixa de treino (extrapolação) e discuta quando faz sentido usar IA em Física.

**07 — PyTorch e a derivada automática**
Tensores; autograd, conferindo que o gradiente bate com o calculado à mão no Módulo 04; a rede do Módulo 05 refeita em PyTorch; otimizadores (SGD e Adam). Mensagem central: o framework faz o que o aluno já fez à mão, só que em escala.

**08 — Linguagem 1: texto vira número**
Caracteres, tokens e vocabulário; modelo de bigramas por contagem num texto em inglês; probabilidade condicional; geração de texto por amostragem; temperatura; perplexidade explicada de forma intuitiva.

**09 — Linguagem 2: um mini modelo neural que escreve em inglês**
Janela de contexto; embeddings (vetores que representam caracteres) e sua visualização em 2D; modelo de caracteres com camadas densas em PyTorch, no estilo de Bengio et al. (2003); treino, geração de texto e comparação com os bigramas.

**10 — Linguagem 3: atenção e um Transformer em miniatura**
Por que o contexto importa; autoatenção (self-attention) com Q, K e V num exemplo minúsculo, calculado passo a passo; um Transformer pequeno de caracteres treinado em CPU por uns 2 minutos; geração de texto. Fechamento: como modelos como ChatGPT e Claude são criados (pré-treinamento, ajuste fino, aprendizado por reforço com feedback humano), seus limites (como as alucinações) e sugestões de próximos passos de estudo.

**Bônus** (só se sobrar crédito e o aluno pedir)
B1: um pêndulo, com uma rede que aprende a dinâmica e o erro que se acumula com o tempo. B2: redes convolucionais (CNN) para imagens. B3: um projeto final escolhido pelo aluno.

## PROGRESSO.md

Para cada módulo concluído, no máximo 10 linhas: data, arquivos criados, conceitos cobertos, funções e nomes que outros módulos podem reaproveitar, dados usados e observações para o próximo módulo.

## Publicação

1. Faça commit com mensagem em português (exemplo: "Módulo 03: regressão logística").
2. Faça push direto para a branch `main`. O aluno autoriza e não usa pull requests neste repositório. Se o push na `main` falhar, envie a sua branch e avise o aluno para criar o PR e fazer o merge.
3. Responda em até 5 linhas: o que foi criado, o link do Colab do módulo e qualquer pendência.
