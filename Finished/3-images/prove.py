import os
from collections import defaultdict

def contar_extensiones(directorio):
    """
    Recorre un directorio y sus subcarpetas para contar archivos por extensión.
    """
    conteo_extensiones = defaultdict(int)

    for root, _, files in os.walk(directorio):
        for file in files:
            # Obtener la extensión del archivo
            _, extension = os.path.splitext(file)
            # Convertir la extensión a minúsculas y contar
            conteo_extensiones[extension.lower()] += 1

    return conteo_extensiones

# Ruta al directorio base
directorio_base = r"C:\Users\maria\Development\ia\Finished\3-images\assets\augmented_images"

# Obtener conteo de extensiones
conteo = contar_extensiones(directorio_base)

# Mostrar resultados
print("Conteo de archivos por extensión:")
for extension, cantidad in conteo.items():
    print(f"{extension if extension else 'Sin extensión'}: {cantidad}")
