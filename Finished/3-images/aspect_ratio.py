import os
from PIL import Image
import statistics
from fractions import Fraction

def calculate_aspect_ratios(directory):
    """
    Calcula las relaciones de aspecto (ancho/alto) de todas las imágenes en un directorio.
    Devuelve un diccionario con estadísticas: promedio, media, mediana y moda.
    """
    aspect_ratios = []

    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                if height != 0:  # Evitar división por cero
                    aspect_ratios.append(Fraction(width, height).limit_denominator())
        except Exception as e:
            print(f"Error leyendo archivo {file}: {e}")

    if not aspect_ratios:
        return None

    # Cálculo de estadísticas
    aspect_ratios_float = [float(r) for r in aspect_ratios]
    return {
        "promedio": Fraction(sum(aspect_ratios_float) / len(aspect_ratios_float)).limit_denominator(),
        "media": Fraction(statistics.mean(aspect_ratios_float)).limit_denominator(),
        "mediana": statistics.median(aspect_ratios),
        "moda": statistics.mode(aspect_ratios) if len(aspect_ratios) > 1 else "Sin moda",
    }

def main():
    base_path = r"C:\\Users\\maria\\Development\\ia\\Finished\\3-images\\assets\\3_extracted_cars"
    
    # Listar las carpetas
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    results = {}

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        stats = calculate_aspect_ratios(folder_path)
        if stats:
            results[folder] = stats

    # Mostrar resultados por carpeta
    print("Estadísticas por carpeta:")
    for folder, stats in results.items():
        print(f"\nCarpeta: {folder}")
        for key, value in stats.items():
            print(f"  {key.capitalize()}: {value}")

    # Determinar la mejor relación de aspecto promedio
    best_folder = max(results, key=lambda x: results[x]["promedio"])
    print("\nMejor relación de aspecto promedio:")
    print(f"  Carpeta: {best_folder}")
    print(f"  Promedio: {results[best_folder]['promedio']}")

if __name__ == "__main__":
    main()
