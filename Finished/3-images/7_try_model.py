import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo de clasificación
model_path = './ModelThree.keras'  # Asegúrate de que sea el modelo correcto
model = load_model(model_path)

# Mapeo de clases
class_labels = {
    0: "nissan_z",
    1: "odyssey",
    2: "prowler",
    3: "ram",
    4: "vocho"
}

def preprocess_image(image_path, target_size=(224, 168)):
    """
    Preprocesa una imagen para usarla en el modelo de clasificación.
    """
    # Leer la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Redimensionar al tamaño esperado
    resized_image = cv2.resize(image, target_size)

    # Normalizar valores de píxeles
    normalized_image = resized_image / 255.0

    # Expandir dimensiones para el modelo
    return np.expand_dims(normalized_image, axis=0)

def predict_image(image_path):
    """
    Clasifica una imagen completa.
    """
    try:
        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image_path)

        # Predecir la clase con el modelo
        predictions = model.predict(preprocessed_image)

        # Obtener clase y confianza
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Etiqueta de clase
        class_label = class_labels.get(predicted_class, "Clase desconocida")

        return class_label, confidence
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return None, None

def classify_images_in_directory(directory_path):
    """
    Clasifica todas las imágenes en un directorio y sus subdirectorios.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                folder_name = os.path.basename(root)

                # Realizar la predicción
                label, confidence = predict_image(image_path)

                # Mostrar los resultados
                if label:
                    print(f"Carpeta: {folder_name}, Imagen: {file}, Predicción: {label}, Confianza: {confidence:.2f}")
                else:
                    print(f"Carpeta: {folder_name}, Imagen: {file}, No se pudo clasificar.")

# Ejemplo de uso
directory_path = "./Test/5"  # Ruta al directorio raíz
classify_images_in_directory(directory_path)
