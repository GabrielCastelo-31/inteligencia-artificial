import random
import matplotlib.pyplot as plt

STEP_SIZE = 0.01  # escala entre pontos para buscar
MAX_STEPS = 100000  # numero de pontos a serem avaliados
INTERVAL_SIZE = 20  # intervalo de busca
MAX_RANDOM_RESTARTS = 5  # numero de tentativas com pontos iniciais aleatorios


# Função para maximizar


def func_to_max(x) -> int:
    return -1*x**2 + 4*x + 4

# Função para minimizar


def func_to_min(x) -> int:
    return -10*x**4 + 5*x**6 + 1

# Algoritmo Hill Climbing


def hill_climb(start, min: bool = False) -> int | bool:
    current_x = start
    # Avalia o valor inicial e escolhe a função correta com base no objetivo
    current_y = func_to_min(current_x) if min else func_to_max(current_x)

    count = 0  # Contador de passos

    # Loop até que não haja mais melhorias
    while True:
        if count >= MAX_STEPS:
            print("Número máximo de passos atingido.")
            return None, None
        next_x = None
        next_y = current_y
        # Gera vizinhos e avalia
        for candidate in linspace(current_x):
            if min:
                # Avalia a função de minimização
                value = func_to_min(candidate)
                if value < current_y:
                    # Atualiza se encontrar um valor melhor
                    next_x = candidate
                    next_y = value
            else:
                # Avalia a função de maximização
                value = func_to_max(candidate)
                if value > current_y:
                    # Atualiza se encontrar um valor melhor
                    next_x = candidate
                    next_y = value
        # Se nenhum vizinho for melhor, termina
        if next_x is None:
            break
        # Move para o próximo ponto
        current_x = next_x
        current_y = next_y

        count += 1  # Incrementa o contador de passos
    return current_x, current_y


def linspace(x) -> list[int]:
    # Gera uma lista de pontos ao redor de x
    points = []
    for i in range(-INTERVAL_SIZE, INTERVAL_SIZE + 1):
        points.append(x + i * STEP_SIZE)
    return points


def random_restart_hill_climbing(user_input=2) -> int:
    """ Executa o algoritmo de Hill Climbing com reinícios aleatórios.
    Retorna o melhor resultado encontrado.

    """
    best_y = None
    best_x = None
    best_iteration = None

    if user_input == 1:
        print("Iniciando Hill Climbing para minimização...")
    else:
        print("Iniciando Hill Climbing para maximização...")
    for i in range(MAX_RANDOM_RESTARTS):
        start = random.random() * 100 - 50  # ponto inicial aleatório entre -50 e 50
        print('Ponto inicial: ', start)

        ########### Minimização ############
        if user_input == 1:
            x, y = hill_climb(start, min=True)
            best_y = y if best_y is None else min(best_y, y)
            best_x = x if best_x is None else (x if y == best_y else best_x)

        else:
            ########### Maximização ############
            x, y = hill_climb(start, min=False)
            best_y = y if best_y is None else max(best_y, y)
            best_x = x if best_x is None else (x if y == best_y else best_x)

        best_iteration = i if best_iteration is None else (
            i if y == best_y else best_iteration)
        print('iteração ', i, ', x: ', x, ' ', 'y: ', y)
    return best_x, best_y, best_iteration


# print(linspace(0))
user_input = int(input(
    "Hill Climbing com reinícios aleatórios. Digite 1 para minimizar ou 2 para maximizar: "))

best_x, best_y, best_iteration = random_restart_hill_climbing(user_input)
if best_x is not None and best_y is not None:

    print(
        f'Melhor resultado encontrado: x = {best_x}, y = {best_y} na iteração {best_iteration}')
else:
    print('Nenhum resultado encontrado.')
