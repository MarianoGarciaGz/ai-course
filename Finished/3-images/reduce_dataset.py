import os
import random
import shutil

# Directorio original
original_dir = './assets/4_data_augmented'
# Directorio reducido
reduced_dir = './assets/5_data_reduced'

# Número máximo de imágenes por clase
max_images_per_class = 4700

# Crear el directorio reducido si no existe
os.makedirs(reduced_dir, exist_ok=True)

for class_name in os.listdir(original_dir):
    class_path = os.path.join(original_dir, class_name)
    if os.path.isdir(class_path):
        # Crear la subcarpeta en el directorio reducido
        reduced_class_path = os.path.join(reduced_dir, class_name)
        os.makedirs(reduced_class_path, exist_ok=True)

        # Obtener todos los archivos de la clase
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        # Seleccionar un subconjunto aleatorio
        selected_files = random.sample(files, min(len(files), max_images_per_class))

        # Copiar los archivos seleccionados al directorio reducido
        for file in selected_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(reduced_class_path, file))

print(f"Dataset reducido guardado en: {reduced_dir}")
