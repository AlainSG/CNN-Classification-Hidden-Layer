import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. Modelo con Sequential (una sola salida: clasificación) ---

base_model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu', name='embedding'),  # Embedding intermedio
    layers.Dense(10, activation='softmax', name='classification')  # Clasificación
])

# --- 2. Compilar y entrenar normalmente (solo clasificación) ---

base_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# Datos simulados
num_train = 1000
num_test = 200
x_train = np.random.rand(num_train, 64, 64, 3).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, num_train), num_classes=10)

x_test = np.random.rand(num_test, 64, 64, 3).astype(np.float32)
y_test = tf.keras.utils.to_categorical(np.random.randint(0, 10, num_test), num_classes=10)

# Entrenamiento
print("Entrenando...")
base_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# --- 3. Modelo auxiliar para extraer embeddings ---

# Creamos un nuevo modelo desde la entrada hasta la capa 'embedding'
embedding_model = models.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('embedding').output
)

# --- 4. Usar el modelo de embedding después del entrenamiento ---

sample_input = np.random.rand(1, 64, 64, 3).astype(np.float32)

embedding_val = embedding_model.predict(sample_input)
prediction_val = base_model.predict(sample_input)

print("Embedding (primeros 10 valores):", embedding_val[0, :10])
print("Predicción (softmax):", prediction_val)
print("Clase predicha:", np.argmax(prediction_val))
