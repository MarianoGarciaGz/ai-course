import os
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model

# Cargar el modelo de clasificación
model_path = './ModelThree.keras'
model = load_model(model_path)

# Mapeo de clases
class_labels = {
    0: "nissan_z",
    1: "odyssey",
    2: "prowler",
    3: "ram",
    4: "vocho"
}

# Cargar el modelo YOLOv5 preentrenado
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Directorio de entrada y salida
input_dir = "./Test/5"
output_dir = "./CroppedImages"
os.makedirs(output_dir, exist_ok=True)

# Padding (proporción del tamaño del recorte, e.g., 0.1 = 10%)
padding_ratio = 0.1  # Ajusta esta variable según tus necesidades

# Lista para almacenar resultados
results = []


def detect_and_crop(image_path, padding_ratio=0.2):
    """
    Detecta el auto más grande en una imagen y recorta la región detectada.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Realizar detección con YOLO
    results = yolo_model(image)
    detections = results.xyxy[0].numpy()

    largest_area = 0
    largest_box = None

    # Buscar el auto o camioneta más grande
    for detection in detections:
        class_id = int(detection[5])
        class_name = results.names[class_id]
        if class_name in ["car", "truck"]:  # Detectar autos y camionetas
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            width = xmax - xmin
            height = ymax - ymin
            area = width * height

            if area > largest_area:
                largest_area = area
                largest_box = (xmin, ymin, xmax, ymax)

    if largest_box is None:
        return None

    # Aplicar padding
    img_height, img_width, _ = image.shape
    xmin, ymin, xmax, ymax = largest_box
    padding_x = int((xmax - xmin) * padding_ratio)
    padding_y = int((ymax - ymin) * padding_ratio)

    xmin = max(0, xmin - padding_x)
    ymin = max(0, ymin - padding_y)
    xmax = min(img_width, xmax + padding_x)
    ymax = min(img_height, ymax + padding_y)

    cropped_image = image[ymin:ymax, xmin:xmax]

    return cropped_image


def preprocess_image(image, target_size=(224, 168)):
    """
    Preprocesa una imagen recortada para usarla en el modelo de clasificación.
    """
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)


def predict_and_save(image_path, output_dir, folder_name, padding_ratio=0.2):
    """
    Recorta, clasifica y guarda las imágenes, y almacena los resultados.
    """
    try:
        cropped_image = detect_and_crop(image_path, padding_ratio)
        if cropped_image is not None:
            # Realizar la predicción
            preprocessed_image = preprocess_image(cropped_image)
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            class_label = class_labels.get(predicted_class, "Clase desconocida")

            # Guardar el recorte en la carpeta correspondiente
            class_folder = os.path.join(output_dir, folder_name)
            os.makedirs(class_folder, exist_ok=True)
            output_path = os.path.join(class_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, cropped_image)

            # Agregar resultado a la lista
            results.append({
                "Carpeta": folder_name,
                "Imagen": os.path.basename(image_path),
                "Predicción": class_label,
                "Confianza": round(confidence, 2)
            })
        else:
            results.append({
                "Carpeta": folder_name,
                "Imagen": os.path.basename(image_path),
                "Predicción": "No detectado",
                "Confianza": 0.0
            })
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")


def process_images_in_directory(input_dir, output_dir, padding_ratio=0.2):
    """
    Procesa todas las imágenes en un directorio y sus subdirectorios.
    """
    for root, _, files in os.walk(input_dir):
        folder_name = os.path.basename(root)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                predict_and_save(image_path, output_dir, folder_name, padding_ratio)

    # Mostrar resultados al final
    print("\nResultados finales:")
    for result in results:
        print(f"{result['Carpeta']} = Predicción: {result['Predicción']}, Confianza: {result['Confianza']:.2f} \n"
              f"{result['Imagen']}\n"
              f"--------------------------------------------------------------------------------------------------")


# Ejecutar el procesamiento
process_images_in_directory(input_dir, output_dir, padding_ratio)
