"""
@author: Gabriel Castelo
Autoencoder convolucional simples usando TensorFlow e Keras
para reconstruir imagens do conjunto CIFAR-10 (aprendizado NÃO supervisionado).
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt


# 1. Carregar e pré-processar CIFAR-10

(train_images, _), (test_images, _) = datasets.cifar10.load_data()

# Normalizar para [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# print("Shape train_images:", train_images.shape)
# print("Shape test_images:", test_images.shape)


# 2. Definir o Autoencoder

# Entrada: mesma shape das imagens
input_img = layers.Input(shape=(32, 32, 3))

# ----- ENCODER -----
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)   # 32x32 -> 16x16

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)   # 16x16 -> 8x8

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  # 8x8 -> 4x4


# ----- DECODER -----
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)   # 4x4 -> 8x8

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)   # 8x8 -> 16x16

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)   # 16x16 -> 32x32

# Saída: mesma shape da imagem, com ativação sigmoid (valores entre 0 e 1)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Modelo completo: input -> decoded
autoencoder = Model(input_img, decoded, name="autoencoder_cifar10")

autoencoder.summary()

# 3. Compilar o modelo
# Como alvo é a própria imagem (x é quase x̂), loss erro medio quadratico funciona bem
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# 4. Treinar o autoencoder
history_ae = autoencoder.fit(
    train_images, train_images,  # x = imagens, y = mesmas imagens
    epochs=10,  # com 20 epochs fica melhor, mas demorou bastante
    batch_size=128,
    shuffle=True,
    validation_data=(test_images, test_images)
)

# 5. Plotar curvas de loss (treino vs validação)
plt.figure(figsize=(8, 4))
plt.plot(history_ae.history['loss'], label='Treino')
plt.plot(history_ae.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.title('Curva de Loss do Autoencoder')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 6. Visualizar reconstruções
n = 10  # número de imagens para visualizar
test_subset = test_images[:n]
decoded_imgs = autoencoder.predict(test_subset)

plt.figure(figsize=(20, 4))
for i in range(n):
    # Imagens originais
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_subset[i])
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    # Imagens reconstruídas
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstruída")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()
