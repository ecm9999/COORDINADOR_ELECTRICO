import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc, precision_score, recall_score, accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo .txt
data = np.loadtxt('ECG5000_TEST.txt')

# Asegurarnos de que los datos estén en una dimensión adecuada
data = data.flatten() if len(data.shape) > 1 else data

# Normalización de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)  # Convertir a formato columna para normalizar
data_normalized = scaler.fit_transform(data)

# Crear ventanas deslizantes para el modelo
window_size = 3  # Tamaño de la ventana de tiempo
X = []
y = []

for i in range(len(data_normalized) - window_size):
    X.append(data_normalized[i:i+window_size].flatten())  # Ventana de entrada
    y.append(data_normalized[i+window_size])  # Siguiente valor como salida

# Convertir a arrays numpy
X = np.array(X)
y = np.array(y)

# Ajustar la forma de X para LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))  # n_muestras, window_size, n_features

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Definir el modelo LSTM
model = Sequential()
model.add(Input(name="serie", shape=(window_size, 1)))  # window_size = 3, n_features = 1
model.add(LSTM(250, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Graficar pérdida de entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación', color='orange')
plt.title('Pérdida Durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.savefig('perdida_entrenamiento.png')  # Guardar el gráfico como PNG
plt.close()

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Validation loss: {loss}')

# Realizar predicciones
predictions = model.predict(X_test)

# Calcular métricas adicionales
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f'R²: {r2}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Calcular el error de reconstrucción
errors = np.mean(np.square(X_test.reshape(X_test.shape[0], X_test.shape[1]) - predictions), axis=1)

# Graficar el histograma del MSE
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='blue', alpha=0.7, label='Normal')
plt.axvline(x=0.011, color='red', linestyle='--', label='Threshold - 0.011')  # Usando un umbral hipotético
plt.title('Histograma del Error Cuadrático Medio (MSE)')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('histograma_mse.png')  # Guardar el gráfico como PNG
plt.close()

# Generar etiquetas (0 = normal, 1 = anómalo) para fines de ejemplo
threshold = 0.011  # Umbral para clasificar anomalías
labels = (errors > threshold).astype(int)  # Etiquetas simuladas según el umbral

# Graficar la curva ROC
fpr, tpr, _ = roc_curve(labels, errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Chance')
plt.title('Curva ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('curva_roc.png')  # Guardar el gráfico como PNG
plt.close()

# Cálculo de métricas de clasificación
predictions_labels = (errors > threshold).astype(int)  # Predicción basada en el umbral
accuracy = accuracy_score(labels, predictions_labels)
precision = precision_score(labels, predictions_labels)
recall = recall_score(labels, predictions_labels)

# Mostrar métricas
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Generar gráfico de serie de tiempo con anomalía
for i in range(len(labels)):
    if labels[i] == 1:  # Si se detecta una anomalía
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(X_test[i].flatten())), X_test[i].flatten(), label='Serie de tiempo')
        plt.title('Serie de Tiempo con Anomalía Detectada')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor Normalizado')
        plt.legend()
        plt.savefig(f'anomalia_{i}.png')  # Guardar el gráfico con un nombre único
        plt.close()
        break  # Solo guardar el primer caso con anomalía
