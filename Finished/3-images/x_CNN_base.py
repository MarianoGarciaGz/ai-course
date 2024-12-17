import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns

# Path to the augmented dataset
dirname = os.path.join(os.getcwd(), './assets/5_data_reduced')
imgpath = dirname + os.sep 

images = []
labels = []
directories = []

print("Reading images from ", imgpath)

# Read images from the dataset
for root, dirnames, filenames in os.walk(imgpath):
    if root == imgpath:  # Skip the base directory
        continue
    
    class_name = os.path.basename(root)
    
    # Check if this is a new class
    if class_name not in directories:
        directories.append(class_name)
    
    class_index = directories.index(class_name)
    
    for filename in filenames:
        if re.search("\.(jpeg)$", filename):
            filepath = os.path.join(root, filename)
            
            try:
                image = plt.imread(filepath)
                
                # Ensure the image has 3 color channels
                if len(image.shape) == 3:
                    images.append(image)
                    labels.append(class_index)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

print('Directories read:', len(directories))
# Convert labels to a NumPy array for boolean indexing
labels = np.array(labels)

# Print the number of images in each directory
print("Images in each directory:", [np.sum(labels == i) for i in range(len(directories))])
print('Total images:', len(images))

# Map class names
sriesgos = directories
print("Classes:", sriesgos)

# Convert images and labels to numpy arrays
X = np.array(images, dtype=np.uint8)
y = np.array(labels)

# Normalize images
X = X.astype('float32') / 255.

# Convert labels to one-hot encoding
y_one_hot = to_categorical(y)

# Dividir el dataset en entrenamiento, validación y prueba
train_X, test_X, train_Y, test_Y = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

print("Training data shape:", train_X.shape)
print("Validation data shape:", valid_X.shape)
print("Testing data shape:", test_X.shape)

# Variables de configuración
INIT_LR = 1e-3
epochs = 14
batch_size = 32
input_shape = train_X.shape[1:]

# Construcción del modelo
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

    Dense(len(sriesgos), activation='softmax')  # Salida igual al número de clases
])

riesgo_model.summary()

# Compilación del modelo
riesgo_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)

# Callbacks para el entrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Entrenamiento del modelo
riesgo_train = riesgo_model.fit(
    train_X, train_Y,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(valid_X, valid_Y),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluación del modelo
test_eval = riesgo_model.evaluate(test_X, test_Y, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Visualización de curvas de entrenamiento
accuracy = riesgo_train.history['accuracy']
val_accuracy = riesgo_train.history['val_accuracy']
loss = riesgo_train.history['loss']
val_loss = riesgo_train.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Matriz de confusión
predicted_classes = np.argmax(riesgo_model.predict(test_X), axis=1)
true_classes = np.argmax(test_Y, axis=1)

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sriesgos, yticklabels=sriesgos)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Reporte de clasificación
print(classification_report(true_classes, predicted_classes, target_names=sriesgos))

# Guardar el modelo
riesgo_model.save("ModelDensoFinal.keras")
