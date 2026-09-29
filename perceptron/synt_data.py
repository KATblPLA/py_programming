from sklearn.neural_network import MLPClassifier
import numpy as np

# Входные данные (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Целевые значения
y = np.array([0, 1, 1, 0])

# Создаем многослойный персептрон:
# hidden_layer_sizes=(2,) — один скрытый слой с 2 нейронами
# max_iter=1000 — количество эпох обучения
clf = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Обучаем сеть
clf.fit(X, y)

# Проверяем результат
predictions = clf.predict(X)
print("Предсказания:", predictions)