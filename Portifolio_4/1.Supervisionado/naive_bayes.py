"""
@author: Gabriel Castelo
Algortitmo de aprendizado supervisionado utilizando Naive Bayes no dataset Iris.
O dataset Iris contem 3 classes de flores (Setosa, Versicolor, Virginica).
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o dataset Iris
dados = load_iris()
X = dados.data      # atributos (4 medidas das flores)
y = dados.target    # rótulos (0,1,2 para as 3 espécies)

# Dividir em treino e teste
# 30% dos dados para teste
# random_state fixa a semente para reproduzir o resultado
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Criar o modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Treinar o modelo
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_teste)

# Avaliar o modelo
acuracia = accuracy_score(y_teste, y_pred)
print("Acurácia no teste:", acuracia)

print("\nRelatório de classificação:")
print(classification_report(y_teste, y_pred, target_names=dados.target_names))

print("Matriz de confusão:")
print(confusion_matrix(y_teste, y_pred))

# teste de previsão com novos dados
novos_dados = [[5.1, 3.5, 1.4, 0.2],   # Provavelmente Setosa
               [6.0, 2.9, 4.5, 1.5],   # Provavelmente Versicolor
               [7.2, 3.6, 6.1, 2.5]]  # Provavelmente Virginica
previsoes_novos = modelo.predict(novos_dados)
for i, pred in enumerate(previsoes_novos):
    print(f"Dado {novos_dados[i]} previsto como: {dados.target_names[pred]}")
