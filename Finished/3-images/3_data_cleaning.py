import os
import cv2
import shutil
import torch

# Directorio con imágenes originales
dataset_dir = "./assets/2_no_duplicated_images"
# Directorio para imágenes que no contienen autos
removed_dir = "./assets/3_removed_images"
# Directorio para guardar áreas de autos detectados
cars_dir = "./assets/3_extracted_cars"

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Crear carpetas de salida si no existen
os.makedirs(removed_dir, exist_ok=True)
os.makedirs(cars_dir, exist_ok=True)

def detect_and_extract_largest_car(image_path, margin=0.2):
    """
    Detecta el auto de mayor tamaño en una imagen usando YOLOv5 y agrega un margen del 20%.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return False, None

    # Realizar detección de objetos con YOLOv5
    results = model(image)
    detections = results.xyxy[0].numpy()

    largest_area = 0
    largest_car = None

    # Filtrar por clase "car" (según el dataset COCO de YOLO)
    for detection in detections:
        class_id = int(detection[5])
        class_name = results.names[class_id]
        if class_name == "car":
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            width = xmax - xmin
            height = ymax - ymin
            area = width * height

            # Buscar el auto con el área más grande
            if area > largest_area:
                largest_area = area

                # Expandir el bounding box con un margen del 20%
                margin_x = int(width * margin)
                margin_y = int(height * margin)

                expanded_xmin = max(0, xmin - margin_x)
                expanded_ymin = max(0, ymin - margin_y)
                expanded_xmax = min(image.shape[1], xmax + margin_x)
                expanded_ymax = min(image.shape[0], ymax + margin_y)

                # Recortar la imagen con el margen incluido
                largest_car = image[expanded_ymin:expanded_ymax, expanded_xmin:expanded_xmax]

    return largest_car is not None, largest_car

# Recorrer todas las imágenes en el dataset
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            file_path = os.path.join(root, file)

            # Ruta relativa para conservar la estructura
            relative_path = os.path.relpath(root, dataset_dir)

            # Directorios de salida con estructura preservada
            removed_subdir = os.path.join(removed_dir, relative_path)
            cars_subdir = os.path.join(cars_dir, relative_path)  # Carpeta específica para autos extraídos
            os.makedirs(removed_subdir, exist_ok=True)
            os.makedirs(cars_subdir, exist_ok=True)

            # Detectar y extraer el auto más grande
            detected, largest_car = detect_and_extract_largest_car(file_path, margin=0.2)
            if detected:
                # Guardar el área del auto más grande
                output_car_path = os.path.join(cars_subdir, f"{os.path.splitext(file)[0]}_largest_car.jpg")
                cv2.imwrite(output_car_path, largest_car)
                print(f"Área de auto más grande extraída: {output_car_path}")
            else:
                # Mover la imagen al subdirectorio de eliminados
                shutil.move(file_path, os.path.join(removed_subdir, file))
                print(f"No se detectó auto: {file} -> Movido a {removed_subdir}")

print("Extracción del auto más grande y limpieza completada.")
