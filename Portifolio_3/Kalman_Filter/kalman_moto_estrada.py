""" kalman_moto_estrada.py
Filtro de Kalman para moto em estrada (1D: posição e velocidade)
Autor: Gabriel Henrique Castelo Costa

Estado: x = [pos; vel]
Dinâmica: x_k = F x_{k-1} + G a_{k-1} + w,   w ~ N(0, Q)
Medição: z_k = H x_k + v,                    v ~ N(0, R)

F = [[1, dt],
     [0,  1]]
G = [[0.5*dt^2],
     [    dt  ]]

Observamos apenas posição via "GPS" (ruidoso).
Opcional: simular aceleração como entrada (IMU) com ruído.
 """
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Configuração ---------------------------
np.random.seed(42)

# Tempo
dt = 0.2               # passo (s)
T = 60.0              # duração total (s)
N = int(T / dt)       # nº de passos

# "Cenário" da moto (trajetória verdadeira)
pos0 = 0.0             # posição inicial (m)
vel0 = 22.0            # velocidade inicial (m/s) ~ 79,2 km/h

# Perfil de aceleração verdadeiro (trocas de marcha / tráfego)


def accel_profile(k: int) -> float:
    t = k * dt
    if t < 10:       # acelera suave
        return 0.8
    elif t < 20:     # mantém
        return 0.0
    elif t < 30:     # freia levemente
        return -0.6
    elif t < 40:     # acelera forte
        return 1.5
    elif t < 50:     # mantém
        return 0.0
    else:            # desacelera leve
        return -0.4


# Ruídos (ajuste conforme seu caso)
sigma_a_true = 0.2     # desvio-padrão do ruído de aceleração REAL (m/s^2)
sigma_gps = 6.0    # ruído do GPS (m) (R)
sigma_a_model = 0.6    # quão incerto consideramos o modelo (para Q)
use_imu_input = True   # se True, filtro usa aceleração "medida" como entrada

# --------------------------- Modelo KF ---------------------------
F = np.array([[1.0, dt],
              [0.0, 1.0]])  # Matriz de transição de estado

G = np.array([[0.5 * dt**2],
              [dt]])  # Matriz de controle de entrada

H = np.array([[1.0, 0.0]])     # Matriz de observação
R = np.array([[sigma_gps**2]])  # Matriz de covariância do ruído de medição v_k

# Q pelo "modelo de aceleração branca" (Singer simplificado)
q = sigma_a_model**2
Q = np.array([[0.25*dt**4, 0.5*dt**3],
              # Matriz de covariância do ruído de processo w_k
              [0.5*dt**3,      dt**2]]) * q

# --------------------- Simulação do "mundo real" ---------------------
true_pos = np.zeros(N)
true_vel = np.zeros(N)
true_pos[0], true_vel[0] = pos0, vel0

acc_true = np.zeros(N)   # aceleração verdadeira
for k in range(1, N):
    a_cmd = accel_profile(k-1)
    # ruído da dinâmica real
    a_real = a_cmd + np.random.normal(0.0, sigma_a_true)
    acc_true[k-1] = a_real
    # integração exata do modelo cinemático 1D
    true_pos[k] = true_pos[k-1] + true_vel[k-1]*dt + 0.5*a_real*dt**2
    true_vel[k] = true_vel[k-1] + a_real*dt
acc_true[-1] = accel_profile(N-1)  # último ponto apenas informativo

# Medições de GPS (posição)
z = true_pos + np.random.normal(0.0, sigma_gps, size=N)

# IMU: aceleração medida com ruído adicional
sigma_imu = 0.5
imu_acc = acc_true + np.random.normal(0.0, sigma_imu, size=N)

# ------------------------ Filtro de Kalman ------------------------
x_hat = np.zeros((2, N))               # estimativa a posteriori
P = np.diag([50.0, 25.0])          # covariância inicial (incerteza maior)
# Inicialização: usar primeira medição como palpite de posição; vel ~ 0
x_hat[:, 0] = np.array([z[0], 0.0])

# buffers para análise
K_hist = np.zeros((N, 2))   # ganho (coluna única, duas linhas)
resid = np.zeros(N)        # inovação (z - H x_pred)

for k in range(1, N):
    # Entrada de controle (IMU) ou assume-se zero/aceleração média
    u_km1 = imu_acc[k-1] if use_imu_input else 0.0

    # PREDIÇÃO
    x_pred = F @ x_hat[:, k-1] + (G * u_km1).ravel()
    P_pred = F @ P @ F.T + Q

    # ATUALIZAÇÃO (medição de posição)
    y = z[k] - (H @ x_pred)        # inovação
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)  # ganho de Kalman (2x1)

    x_hat[:, k] = x_pred + (K * y).ravel()
    P = (np.eye(2) - K @ H) @ P_pred

    K_hist[k, :] = K.ravel()
    resid[k] = y

# ------------------------------ Plots ------------------------------
t = np.arange(N) * dt

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.title("Posição da moto (verdade, GPS e Kalman)")
plt.plot(t, true_pos, label="Posição verdadeira", linewidth=2)
plt.scatter(t, z, s=10, alpha=0.5, label="GPS (posição)")
plt.plot(t, x_hat[0], "--", label="Estimativa Kalman (posição)", linewidth=2)
plt.ylabel("Posição (m)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Velocidade da moto (verdade e Kalman)")
plt.plot(t, true_vel, label="Velocidade verdadeira", linewidth=2)
plt.plot(t, x_hat[1], "--",
         label="Estimativa Kalman (velocidade)", linewidth=2)
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Resíduos e ganho (opcional)
plt.figure(figsize=(12, 4))
plt.plot(t, resid, label="Inovação (z - Hx_pred)")
plt.axhline(0, color="k", linewidth=0.8)
plt.title("Resíduo de medição (diagnóstico)")
plt.xlabel("Tempo (s)")
plt.ylabel("metros")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, K_hist[:, 0], label="K_pos")
plt.plot(t, K_hist[:, 1], label="K_vel")
plt.title("Ganho de Kalman ao longo do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("ganho adimensional")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
