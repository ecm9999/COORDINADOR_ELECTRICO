import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import random as rd
import matplotlib.pyplot as plt

# Cargar los datos
datos_combinadosAJAHUEL_H1 = pd.read_csv('data.csv')
y3 = datos_combinadosAJAHUEL_H1['X3'].values

# Preparar los datos
yw = []
yt = []
for i in range(len(y3) - 3):
    row = [y3[i], y3[i+1], y3[i+2]]
    yw.append(row)
    yt.append(y3[i+3])

# Convertir a arrays numpy
yw = np.array(yw)
yt = np.array(yt)

# Dividir en conjunto de entrenamiento y prueba
seed = 12122008
rd.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
yw_train, yw_test, yt_train, yt_test = train_test_split(yw, yt, test_size=0.2, random_state=seed)

# Reestructurar los datos para que sean compatibles con el modelo LSTM
yw_train = yw_train.reshape((yw_train.shape[0], 3, 1))
yw_test = yw_test.reshape((yw_test.shape[0], 3, 1))

# Definir y compilar el modelo LSTM
model = Sequential()
model.add(Input(name="serie", shape=(3, 1)))
model.add(LSTM(350, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(yw_train, yt_train, epochs=10, validation_data=(yw_test, yt_test))

# Evaluar el modelo
loss = model.evaluate(yw_test, yt_test)
print(f'Validation loss: {loss}')

# Hacer predicciones
predicciones = model.predict(yw_test)

# Graficar las predicciones
plt.figure(figsize=(10,6))
plt.plot(range(len(yt_test)), yt_test, label='Valores Reales')
plt.plot(range(len(predicciones)), predicciones, color='red', label='Predicciones')
plt.legend()
plt.title('Predicciones del modelo LSTM')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Valor')
plt.savefig('predicciones.png')
plt.show()
