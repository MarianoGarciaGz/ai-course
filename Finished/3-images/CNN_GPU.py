import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

# Opcional: Mixed Precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Ruta al dataset aumentado
dirname = os.path.join(os.getcwd(), './assets/5_data_reduced')
batch_size = 16  # Usa un batch más pequeño para ahorrar memoria
img_height = 224
img_width = 224

# Generadores de datos para entrenamiento y validación
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

train_generator = datagen.flow_from_directory(
    dirname,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Conjunto de entrenamiento
)

valid_generator = datagen.flow_from_directory(
    dirname,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Conjunto de validación
)

# Obtén las clases para el reporte posterior
sriesgos = list(train_generator.class_indices.keys())
num_classes = len(sriesgos)

# Configuración del modelo
INIT_LR = 1e-3
epochs = 20
input_shape = (img_height, img_width, 3)  # definido por target_size

riesgo_model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),

    Flatten(),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),

    Dense(num_classes, activation='softmax', dtype='float32')  # Forzar salida a float32
])

riesgo_model.summary()

# Compilación del modelo
riesgo_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Entrenar usando generadores
riesgo_train = riesgo_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluar el modelo en el conjunto de validación (opcional)
val_loss, val_acc = riesgo_model.evaluate(valid_generator, verbose=1)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_acc)

# Para la matriz de confusión y reporte, primero obtén las predicciones
valid_generator.reset()  # Asegurar que empiece desde el principio
predictions = riesgo_model.predict(valid_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

riesgo_model.save("ModelDensoFinal.keras")
