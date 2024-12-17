import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Directorio con imágenes redimensionadas
directorio_origen = "./assets/4_resized_images"

# Directorio donde se guardarán las imágenes augmentadas
directorio_destino = "./assets/4_data_augmented"

# Crear carpeta destino si no existe
os.makedirs(directorio_destino, exist_ok=True)

# Configuración del Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=5,        # Rotación aleatoria
    width_shift_range=.1,    # Desplazamiento horizontal
    height_shift_range=14,   # Desplazamiento vertical
    shear_range=0.1,          # Transformación de corte
    zoom_range=0.1,           # Zoom aleatorio
    horizontal_flip=False,     # Volteo horizontal
    fill_mode='reflect'       # Relleno de píxeles vacíos
)

# Función para agregar ruido
def agregar_ruido(imagen_array, tipo="gaussiano"):
    if tipo == "gaussiano":
        mean = 0
        stddev = 0.05
        ruido = np.random.normal(mean, stddev, imagen_array.shape)
        imagen_array = imagen_array + ruido
    elif tipo == "salt_pepper":
        prob = 0.02  # Probabilidad de ruido
        ruido = np.random.choice([0, 1, 2], size=imagen_array.shape, p=[prob, 1 - 2 * prob, prob])
        imagen_array[ruido == 0] = 0  # Salt (píxeles negros)
        imagen_array[ruido == 2] = 255  # Pepper (píxeles blancos)
    imagen_array = np.clip(imagen_array, 0, 255)  # Asegurar valores válidos
    return imagen_array

# Función para procesar imágenes
def augmentar_imagenes(carpeta_origen, carpeta_destino, datagen, num_augmented=10, ruido="gaussiano"):
    for raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                ruta_archivo = os.path.join(raiz, archivo)
                
                # Subcarpeta destino (preservar estructura)
                subcarpeta_relativa = os.path.relpath(raiz, carpeta_origen)
                ruta_subcarpeta_destino = os.path.join(carpeta_destino, subcarpeta_relativa)
                os.makedirs(ruta_subcarpeta_destino, exist_ok=True)

                # Cargar imagen
                img = Image.open(ruta_archivo)
                img = img.convert("RGB")  # Asegurarse de que esté en RGB
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                # Agregar ruido
                img_array = agregar_ruido(img_array, tipo=ruido)
                img_array = img_array.reshape((1,) + img_array.shape)  # Agregar batch dimension

                # Generar imágenes augmentadas
                contador = 0
                for batch in datagen.flow(img_array, batch_size=1,
                                          save_to_dir=ruta_subcarpeta_destino,
                                          save_prefix='aug', save_format='jpeg'):
                    contador += 1
                    if contador >= num_augmented:
                        break

# Ejecutar el Data Augmentation con ruido gaussiano
augmentar_imagenes(directorio_origen, directorio_destino, datagen, num_augmented=10, ruido="gaussiano")

print(f"Imágenes augmentadas guardadas en: {directorio_destino}")
