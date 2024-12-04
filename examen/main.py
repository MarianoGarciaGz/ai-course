import numpy as np
import matplotlib.pyplot as plt


# Datos de entrada (características)
X = np.array(
    [
        [0.9, 0.8, 0.2],
        [0.7, 0.6, 0.5],
        [0.4, 0.4, 0.8],
        [0.8, 0.9, 0.3],
        [0.5, 0.7, 0.6],
        [0.3, 0.5, 0.9],
    ]
)

# Etiquetas de salida (categorías)
y = np.array(
    [
        [1, 0, 0],  # Riesgo Bajo
        [0, 1, 0],  # Riesgo Medio
        [0, 0, 1],  # Riesgo Alto
        [1, 0, 0],  # Riesgo Bajo
        [0, 1, 0],  # Riesgo Medio
        [0, 0, 1],  # Riesgo Alto
    ]
)

# Definición de hiperparámetros
input_size = 3  # Número de características de entrada
hidden_size = 4  # Número de neuronas en la capa oculta
output_size = 3  # Número de clases (Riesgo Bajo, Medio, Alto)
learning_rate = 0.1
epochs = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    # Propagación hacia adelante
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Cálculo de la pérdida (Entropía Cruzada Categórica)
    loss = -np.mean(np.sum(y * np.log(A2 + 1e-8), axis=1))

    # Retropropagación
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Actualización de pesos y sesgos
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Impresión de la pérdida cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1}/{epochs}, Pérdida: {loss:.4f}")

# Fijar la relación deuda-ingreso en 0.5
fixed_debt_income_ratio = 0.5

# Crear una malla de puntos
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Generar entradas para la malla
grid_X = np.c_[
    xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, fixed_debt_income_ratio)
]

# Propagación hacia adelante en la malla
Z1_grid = np.dot(grid_X, W1) + b1
A1_grid = sigmoid(Z1_grid)
Z2_grid = np.dot(A1_grid, W2) + b2
A2_grid = softmax(Z2_grid)
predictions_grid = np.argmax(A2_grid, axis=1)

# Redimensionar las predicciones
Z = predictions_grid.reshape(xx.shape)

# Gráfica
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)

# Graficar los puntos de datos originales
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    c=np.argmax(y, axis=1),
    s=100,
    edgecolors="k",
    cmap=plt.cm.coolwarm,
)

# Leyenda
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto"],
)
plt.xlabel("Historial de pagos")
plt.ylabel("Ingresos mensuales")
plt.title("Frontera de decisión con relación deuda-ingreso fija en 0.5")
plt.show()
