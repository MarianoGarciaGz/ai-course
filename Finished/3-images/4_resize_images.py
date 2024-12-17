import os
import cv2

# Directorio con las imágenes originales
directorio_origen = "./assets/3_extracted_cars"

# Directorio donde se guardarán las imágenes redimensionadas
directorio_destino = "./assets/4_resized_images"

# Dimensiones deseadas (ancho x alto)
DIMENSIONES = (224, 168)

# Crear carpeta destino si no existe
os.makedirs(directorio_destino, exist_ok=True)

# Procesar imágenes
for raiz, _, archivos in os.walk(directorio_origen):
    for archivo in archivos:
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            ruta_origen = os.path.join(raiz, archivo)

            # Crear subcarpeta correspondiente en el destino
            subcarpeta_relativa = os.path.relpath(raiz, directorio_origen)
            ruta_destino_carpeta = os.path.join(directorio_destino, subcarpeta_relativa)
            os.makedirs(ruta_destino_carpeta, exist_ok=True)

            # Leer la imagen
            imagen = cv2.imread(ruta_origen)

            if imagen is not None:
                # Redimensionar la imagen a 300x224 sin preservar la relación de aspecto
                imagen_redimensionada = cv2.resize(imagen, DIMENSIONES)

                # Guardar la imagen redimensionada
                ruta_destino = os.path.join(ruta_destino_carpeta, archivo)
                cv2.imwrite(ruta_destino, imagen_redimensionada)
                print(f"Redimensionada: {archivo} -> {ruta_destino}")
            else:
                print(f"Error al leer la imagen: {ruta_origen}")

print(f"Todas las imágenes se han redimensionado y guardado en {directorio_destino}")
