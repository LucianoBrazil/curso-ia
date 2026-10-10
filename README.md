# Inteligência Artificial do Zero

Um curso de inteligência artificial em português, do zero até um Transformer em miniatura, usando Matemática, Física e Inglês como fio condutor.

## Como estudar

1. Na primeira vez, autorize o Colab a acessar o GitHub, porque o repositório é privado: no Colab, abra **Arquivo › Abrir notebook**, vá à aba **GitHub**, marque a opção de incluir repositórios privados ("Include private repos") e entre com a sua conta do GitHub. Depois disso, os links "Abrir no Colab" funcionam.
2. Abra cada módulo pelo link "Abrir no Colab" da tabela abaixo. Não precisa instalar nada.
3. Execute as células em ordem, de cima para baixo, com Shift+Enter.
4. Faça os exercícios da seção "Mão na massa" antes de olhar as soluções.

Sobre os arquivos de dados: os módulos 08 a 10 obtêm sozinhos o texto de *Alice no País das Maravilhas* (do corpus Gutenberg da biblioteca NLTK). No B3, para não esperar uns 10 minutos de treino, baixe o arquivo `dados/b3_pesos.pt` pela página do repositório no GitHub e envie-o pelo painel de arquivos do Colab (o ícone de pasta, à esquerda) antes de rodar o notebook; sem ele, o notebook treina tudo do zero.

## Módulos

| Nº | Módulo | Notebook |
|----|--------|----------|
| 00 | Ferramentas: Python, NumPy e gráficos | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/00-ferramentas.ipynb) |
| 01 | O que é "aprender": ajustando uma reta | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/01-ajustando-uma-reta.ipynb) |
| 02 | Várias variáveis: vetores, matrizes e normalização | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/02-varias-variaveis.ipynb) |
| 03 | Classificação: regressão logística | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/03-regressao-logistica.ipynb) |
| 04 | A primeira rede neural, do zero | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/04-primeira-rede-neural.ipynb) |
| 05 | Reconhecendo dígitos escritos à mão | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/05-digitos.ipynb) |
| 06 | Física + IA: o lançamento oblíquo | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/06-fisica-lancamento.ipynb) |
| 07 | PyTorch e a derivada automática | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/07-pytorch.ipynb) |
| 08 | Linguagem 1: texto vira número | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/08-texto-vira-numero.ipynb) |
| 09 | Linguagem 2: um mini modelo neural que escreve em inglês | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/09-modelo-neural-caracteres.ipynb) |
| 10 | Linguagem 3: atenção e um Transformer em miniatura | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/10-transformer.ipynb) |

## Bônus

| Nº | Módulo | Notebook |
|----|--------|----------|
| B1 | O pêndulo: uma rede que aprende a dinâmica | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B1-pendulo.ipynb) |
| B2 | Redes convolucionais (CNN) para imagens | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B2-redes-convolucionais.ipynb) |
| B3 | Na fronteira: modelos que pensam antes de responder | [Abrir no Colab](https://colab.research.google.com/github/LucianoBrazil/curso-ia/blob/main/modulos/B3-modelos-que-pensam.ipynb) |

Para rodar fora do Colab: `pip install -r requirements.txt` e abra os arquivos da pasta `modulos/` no Jupyter.
