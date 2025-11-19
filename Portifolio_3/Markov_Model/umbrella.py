
"""
HMM (Hidden Markov Model) didático — Clima (oculto) vs. Guarda-chuva (observado)
Autor: Gabriel
Descrição:
  - Implementa: forward, backward, posterior marginals, Viterbi, log-likelihood.
Como executar:
    python3 -m venv env
    source env/bin/activate  # Linux/Mac
    . \env\Scripts\activate   # Windows
    pip install -r requirements.txt
    python umbrella.py
"""
import numpy as np

# ---------------------- Definição do modelo ----------------------
STATES = ["Rainy", "Sunny"]           # 0, 1
OBSYMS = ["Yes", "No"]                # 0, 1

# Priors e matrizes (edite à vontade)
pi = np.array([0.5, 0.5], dtype=float)            # P(X1) a priori
MATRIX_TRANSITION = np.array([[0.7, 0.3],                        # P(Xt|Xt-1) : linhas=estado anterior, colunas=estado atual
                              [0.3, 0.7]], dtype=float)
UMBRELLA_PROBS = np.array([[0.9, 0.1],                        # P(Yt|Xt): linhas=estado, colunas=obs
                           [0.2, 0.8]], dtype=float)

# Sequências de exemplo (0=Yes, 1=No)
obs1 = np.array([0, 0], dtype=int)  # guarda-chuva em ambos os dias
# guarda-chuva, guarda-chuva, não, guarda-chuva, guarda-chuva
obs2 = np.array([0, 0, 1, 0, 0], dtype=int)

# ------------------------- Utilidades -------------------------


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(p.sum())
    return p / s if s > 0 else p


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[idx] = 1.0
    return v

# ---------------------- Forward (filtragem) ----------------------


def forward_scaled(pi: np.ndarray, MATRIX_TRANSITION: np.ndarray, UMBRELLA_PROBS: np.ndarray, obs: np.ndarray):
    """
    Retorna:
      alpha  : (T,N) com P(Xt|y1:t) já normalizado
      scales : (T,) fatores de escala usados (1/soma antes de normalizar)
    """
    T, N = len(obs), len(pi)
    alpha = np.zeros((T, N), dtype=float)
    scales = np.zeros(T, dtype=float)

    # t=0
    alpha[0] = pi * UMBRELLA_PROBS[:, obs[0]]
    scales[0] = 1.0 / alpha[0].sum()
    alpha[0] *= scales[0]

    # t >= 1
    for t in range(1, T):
        # previsão e atualização
        pred = alpha[t-1] @ MATRIX_TRANSITION               # P(Xt | y1:t-1)
        # aplica verossimilhança
        alpha[t] = pred * UMBRELLA_PROBS[:, obs[t]]
        scales[t] = 1.0 / alpha[t].sum()
        alpha[t] *= scales[t]              # normaliza

    return alpha, scales

# ---------------------- Backward (escalonado) ----------------------


def backward_scaled(MATRIX_TRANSITION: np.ndarray, UMBRELLA_PROBS: np.ndarray, obs: np.ndarray, scales: np.ndarray):
    """
    Usa os mesmos 'scales' do forward para manter consistência.
    Retorna beta (T,N) proporcional a P(y_{t+1:T} | Xt) * constante.
    """
    T, N = len(obs), MATRIX_TRANSITION .shape[0]
    beta = np.zeros((T, N), dtype=float)

    beta[-1] = 1.0 * scales[-1]
    for t in range(T-2, -1, -1):
        # beta[t,i] = sum_j MATRIX_TRANSITION [i,j] * UMBRELLA_PROBS[j, y_{t+1}] * beta[t+1,j]
        beta[t] = (MATRIX_TRANSITION  @
                   (UMBRELLA_PROBS[:, obs[t+1]] * beta[t+1])) * scales[t]

    return beta

# ---------------------- Suavização (smoothing) ----------------------


def smooth(pi: np.ndarray, MATRIX_TRANSITION: np.ndarray, UMBRELLA_PROBS: np.ndarray, obs: np.ndarray):
    """
    Combina forward e backward escalonados e retorna:
      gamma: (T,N) com P(Xt | y1:T)
    """
    alpha, scales = forward_scaled(pi, MATRIX_TRANSITION, UMBRELLA_PROBS, obs)
    beta = backward_scaled(MATRIX_TRANSITION, UMBRELLA_PROBS, obs, scales)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma

# ---------------------------- Viterbi ----------------------------


def viterbi(pi: np.ndarray, MATRIX_TRANSITION: np.ndarray, UMBRELLA_PROBS: np.ndarray, obs: np.ndarray):
    """
    Retorna caminho mais provável (lista de estados) e sua probabilidade aproximada.
    """
    T, N = len(obs), len(pi)
    delta = np.zeros((T, N), dtype=float)
    psi = np.zeros((T, N), dtype=int)

    # t=0
    delta[0] = pi * UMBRELLA_PROBS[:, obs[0]]
    delta[0] = normalize(delta[0])

    # t>=1
    for t in range(1, T):
        for j in range(N):
            vals = delta[t-1] * MATRIX_TRANSITION[:, j]
            psi[t, j] = int(np.argmax(vals))
            delta[t, j] = vals[psi[t, j]] * UMBRELLA_PROBS[j, obs[t]]
        delta[t] = normalize(delta[t])

    # backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    # probabilidade (aproximada) do caminho escolhido
    p_hat = float(delta[-1, path[-1]])
    return path.tolist(), p_hat

# ------------------------------ Demo ------------------------------


def run_demo(obs: np.ndarray, label: str):
    print(f"\n=== DEMO {label} | Observações:",
          [OBSYMS[o] for o in obs], "===")

    alpha, _ = forward_scaled(pi, MATRIX_TRANSITION, UMBRELLA_PROBS, obs)
    print("\nFiltragem P(Xt | y1:t):")
    for t, a in enumerate(alpha, 1):
        print(f" t={t}: Rainy={a[0]:.3f} | Sunny={a[1]:.3f}")

    gamma = smooth(pi, MATRIX_TRANSITION, UMBRELLA_PROBS, obs)
    print("\nSuavização P(Xt | y1:T):")
    for t, g in enumerate(gamma, 1):
        print(f" t={t}: Rainy={g[0]:.3f} | Sunny={g[1]:.3f}")

    path, prob = viterbi(pi, MATRIX_TRANSITION, UMBRELLA_PROBS, obs)
    print("\nViterbi (caminho mais provável):", [
          STATES[i] for i in path], f"(aproxP={prob:.4f})")


if __name__ == "__main__":
    run_demo(obs1, "Obs1")
    run_demo(obs2, "Obs2")
