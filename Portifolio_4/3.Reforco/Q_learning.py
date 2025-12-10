"""
@author: Gabriel Castelo
Exemplo simples de Q-learning para aprendizado por reforco em um ambiente de linha com 5 estados e um objetivo.
Como se fosse um robo aprendendo a navegar em um ambiente simples.
"""

import numpy as np

# Exemplo simples de Q-learning com matriz
# Estados: 0, 1, 2, 3, 4, 5
# Estado 5 e o objetivo

n_states = 6
goal_state = 5

# Matriz de recompensas R (estado atual x proximo estado)
# -1 = movimento proibido
R = -1 * np.ones((n_states, n_states))

# Definir movimentos possiveis e recompensas

# Estado 0: pode ir para 1 e 4
R[0, 1] = 0
R[0, 4] = 0

# Estado 1: pode ir para 0, 3 e 5 (5 e o objetivo)
R[1, 0] = 0
R[1, 3] = 0
R[1, 5] = 100

# Estado 2: pode ir para 3
R[2, 3] = 0

# Estado 3: pode ir para 1, 2 e 4
R[3, 1] = 0
R[3, 2] = 0
R[3, 4] = 0

# Estado 4: pode ir para 0, 3 e 5
R[4, 0] = 0
R[4, 3] = 0
R[4, 5] = 100

# Estado 5: objetivo, pode ficar nele
R[5, 5] = 100

# Matriz Q inicial (tudo zero)
Q = np.zeros((n_states, n_states))

# Hiperparametros
gamma = 0.8    # fator de desconto
alpha = 0.1    # taxa de aprendizado
episodes = 3000


def acoes_possiveis(estado):
    """Retorna indices de proximos estados possiveis."""
    return np.where(R[estado] != -1)[0]


# Treino
for _ in range(episodes):
    # Escolhe estado inicial aleatorio
    estado = np.random.randint(0, n_states)

    # Roda ate chegar no objetivo
    while estado != goal_state:
        # Escolhe uma acao aleatoria entre as possiveis
        possiveis = acoes_possiveis(estado)
        acao = np.random.choice(possiveis)
        prox_estado = acao

        recompensa = R[estado, acao]

        # Melhor Q no proximo estado
        possiveis_prox = acoes_possiveis(prox_estado)
        if len(possiveis_prox) > 0:
            melhor_q_prox = np.max(Q[prox_estado, possiveis_prox])
        else:
            melhor_q_prox = 0

        # Atualizacao Q-learning
        Q[estado, acao] = Q[estado, acao] + alpha * (
            recompensa + gamma * melhor_q_prox - Q[estado, acao]
        )

        estado = prox_estado

print("Matriz R (recompensas):")
print(R)

print("\nMatriz Q aprendida:")
print(np.round(Q, 1))

# Funcao para mostrar melhor caminho a partir de um estado


def melhor_rota(estado_inicial):
    estado = estado_inicial
    rota = [estado]
    max_passos = 20

    for _ in range(max_passos):
        if estado == goal_state:
            break
        possiveis = acoes_possiveis(estado)
        q_vals = Q[estado, possiveis]
        melhor = np.argmax(q_vals)
        acao = possiveis[melhor]
        estado = acao
        rota.append(int(estado))
    return rota


print("\nMelhores rotas ate o estado 5:")
for s in range(n_states):
    print(f"Estado {s}: {melhor_rota(s)}")
