from PIL import Image
import os

# Directorio que deseas limpiar
directorio = r"C:\Users\maria\Development\ia\Finished\3-images\cleaned_images"

# Función para verificar si una imagen está rota
def es_imagen_valida(ruta_imagen):
    try:
        with Image.open(ruta_imagen) as img:
            img.verify()  # Verifica si la imagen se puede abrir
        return True
    except Exception as e:
        print(f"Imagen rota: {ruta_imagen} ({e})")
        return False

# Recorrer todas las subcarpetas y archivos
imagenes_eliminadas = 0
for raiz, _, archivos in os.walk(directorio):
    for archivo in archivos:
        ruta_archivo = os.path.join(raiz, archivo)
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            if not es_imagen_valida(ruta_archivo):
                try:
                    os.remove(ruta_archivo)
                    print(f"Eliminada: {ruta_archivo}")
                    imagenes_eliminadas += 1
                except Exception as e:
                    print(f"Error al eliminar {ruta_archivo}: {e}")

print(f"Total de imágenes eliminadas: {imagenes_eliminadas}")
