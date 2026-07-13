# artificial-intelligence

Repository for projects developed in the **FGA0221 – Artificial Intelligence** course of the Software Engineering program at the University of Brasília (UnB), taught by Professor Fabiano Araujo Soares.

## How to Run the Projects

Each project has its own dependencies and requirements. It is recommended to create a separate Python virtual environment for each project using `venv` or `conda`. Then, install the dependencies listed in each project's `requirements.txt` file.

Example using `venv`:

```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
pip install -r requirements.txt
```

To run the scripts, use the following command:

```bash
python portfolio_name/project_folder/file_name.py
```

Example:

```bash
python Portfolio_2/1_informed_search/maze_A_star.py # Runs the A* search project
```

## Project Structure

The projects are organized into three main categories:

1. Classical AI Methods;
2. Handling Uncertainty;
3. Machine Learning.

All projects are provided as commented `.py` files and include an attached document containing an explanation of the problem, usage examples, and explanatory images.

## Classical AI Methods

1. Informed search project;
2. Uninformed search project;
3. Complex search project;
4. Genetic algorithm project;
5. Project using Constraint Satisfaction Problems (CSPs) and their methods to solve a problem;
6. Project using knowledge base concepts to solve a problem.

## Handling Uncertainty

1. Project demonstrating the use of Bayesian networks;
2. Project demonstrating the use of Hidden Markov Models;
3. Project demonstrating the use of Kalman filters.

## Machine Learning

1. Project demonstrating the use of supervised learning methods other than neural networks;
2. Project demonstrating the use of unsupervised learning;
3. Project demonstrating the use of reinforcement learning;
4. Project demonstrating the use of a deep learning architecture. Any type of architecture may be used, such as CNN, RNN, LSTM, or Transformer. Well-known architectures such as VGGNet, AlexNet, or LeNet may also be used. Public datasets should be used for training, such as MNIST or Cats and Dogs. Several suitable datasets can be found on Kaggle.


# inteligencia-artificial (PORTUGUESE PT-BR)
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
