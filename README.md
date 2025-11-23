# inteligencia-artificial
Repositório destinado aos projetos desenvolvidos na disciplina **FGA0221- Inteligência Artificial** do curso de Engenharia de Software da Universidade de Brasília (UnB), ministrada pelo professor Fabiano Araujo Soares.

## Como rodar os projetos
Cada projeto possui suas próprias dependências e requisitos. Recomenda-se criar um ambiente virtual Python para cada projeto utilizando `venv` ou `conda`. Em seguida, instalar as dependências listadas no arquivo `requirements.txt` de cada projeto. Exemplo usando `venv`:

```bash
    python -m venv env
    source env/bin/activate  # No Windows use: env\Scripts\activate
    pip install -r requirements.txt
```
Para rodar os scripts, utilize o comando:

```bash
    python nome_do_portifolio/pasta_do_projeto/nome_do_arquivo.py
```
Exemplo:

```bash
    python Portifolio_2/1_busca_informada/maze_A_star.py # Roda o projeto de busca A*
```

## Estrutura dos Projetos

Os projetos estão organizados em três categorias principais:
1. Métodos Clássicos de IA;
2. Tratando Incerteza;
3. Aprendizado de Máquina.

Todos os projetos são apresentados em arquivo .py comentado e com documento
anexo contendo a explicação do problema e exemplo de uso do programa desenvolvido com
imagens explicativas.

## Métodos Clássicos de IA

1. Projeto de busca informada;
2. Projeto de busca não informada;
3. Projeto de busca complexa;
4. Projeto de algoritmo genético;
5. Projeto utilizando CSPs e seus métodos para solução de um problema;
6. Projeto utilizando os conceitos de banco de conhecimentos para solução de um problema.


## Tratando Incerteza

1. Projeto exemplificando o uso de redes Bayesianas;
2. Projeto exemplificando o uso de modelos Markovianos ocultos;
3. Projeto exemplificando o uso de Filtros de Kalman;

## Aprendizado de Máquina

1. Projeto exemplificando o uso aprendizado supervisionado diferente de redes neurais;
2. Projeto exemplificando o uso de não supervisionado;
3. Projeto exemplificando o uso de aprendizado por reforço;
4. Projeto exemplificando o uso de uma arquitetura Deep Learning. Pode ser qualquer tipo
de arquitetura como CNN, RNN, LSTM, Transformer, etc. E pode se utilizar arquiteturas
conhecidas como VGGNet, AlexNet, LeNet, etc. Use repositórios públicos para os dados
de treinamento (como a MINIST ou o Cats and Dogs. Vários repositórios do tipo podem
ser encontrados no Kaggle.)
