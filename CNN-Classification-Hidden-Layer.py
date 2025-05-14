import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 🔧 1. Crear el modelo base con Sequential
base_model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu', name='embedding'),  # 🧠 Embedding aquí
    layers.Dense(10, activation='softmax', name='classification')  # Clasificación
])

# 📤 2. Crear un modelo con múltiples salidas desde el modelo Sequential
# Usamos el modelo funcional para acceder a capas intermedias
inputs = base_model.input
embedding_output = base_model.get_layer('embedding').output
classification_output = base_model.output

dual_output_model = models.Model(inputs=inputs, outputs=[embedding_output, classification_output])

# 📥 3. Prueba con una imagen aleatoria
sample_input = np.random.rand(1, 64, 64, 3).astype(np.float32)

embedding, prediction = dual_output_model.predict(sample_input)

print("🔹 Embedding shape:", embedding.shape)         # (1, 128)
print("🔹 Predicción (softmax):", prediction.shape)   # (1, 10)
