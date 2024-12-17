import os
import hashlib
import shutil

# Función para calcular el hash MD5 de un archivo
def calcular_hash_md5(archivo):
    hasher = hashlib.md5()
    with open(archivo, 'rb') as f:
        for bloque in iter(lambda: f.read(4096), b""):
            hasher.update(bloque)
    return hasher.hexdigest()

# Directorio de origen (donde están las imágenes)
directorio_origen = r"C:\Users\maria\Development\ia\Finished\3-images\assets\1_scraping\ramram"

# Directorio de destino (donde se guardarán las imágenes únicas)
directorio_destino = r"C:\Users\maria\Development\ia\Finished\3-images\assets\2_no_duplicated_images\ram"

# Crear la carpeta destino si no existe
os.makedirs(directorio_destino, exist_ok=True)

# Recorrer las carpetas padres (vocho, nissan_z, etc.)
for carpeta_padre in os.listdir(directorio_origen):
    ruta_padre = os.path.join(directorio_origen, carpeta_padre)
    
    # Verificar si es una carpeta
    if os.path.isdir(ruta_padre):
        print(f"Procesando carpeta: {carpeta_padre}")
        
        # Crear una carpeta en el destino para la carpeta padre
        carpeta_destino = os.path.join(directorio_destino, carpeta_padre)
        os.makedirs(carpeta_destino, exist_ok=True)

        # Diccionario para almacenar hashes de archivos únicos dentro de la carpeta padre
        hashes_vistos = {}

        # Recorrer subcarpetas y archivos de la carpeta padre
        for raiz, _, archivos in os.walk(ruta_padre):
            for archivo in archivos:
                if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.webp')):
                    ruta_archivo = os.path.join(raiz, archivo)
                    
                    # Calcular el hash MD5 del archivo
                    hash_md5 = calcular_hash_md5(ruta_archivo)
                    
                    if hash_md5 not in hashes_vistos:
                        # Si el hash no está en el diccionario, copia el archivo
                        hashes_vistos[hash_md5] = ruta_archivo
                        
                        # Copiar al directorio destino dentro de la carpeta padre
                        nombre_destino = os.path.join(carpeta_destino, f"{hash_md5[:8]}_{archivo}")
                        shutil.copy2(ruta_archivo, nombre_destino)
                        print(f"Copiado: {archivo}")
                    else:
                        print(f"Duplicado omitido en {carpeta_padre}: {archivo}")

print(f"Imágenes únicas copiadas a: {directorio_destino}")
